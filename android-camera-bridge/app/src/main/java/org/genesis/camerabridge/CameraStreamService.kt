package org.genesis.camerabridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.ImageFormat
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.media.Image
import android.media.ImageReader
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.Log
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.net.InetAddress
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * Android's hardware-facing camera adapter.
 *
 * It converts Camera2 YUV planes to packed RGB and forwards raw frames to
 * the Genesis daemon's optic-capture limb. It performs no recognition,
 * scene analysis, or perception; the daemon publishes frames to the
 * canonical /genesis_optic_nerve mapping consumed by the existing mind.
 */
class CameraStreamService : Service() {
    private companion object {
        const val TAG = "GenesisCamera"
        const val NOTIFICATION_ID = 7
        const val NOTIFICATION_CHANNEL = "genesis_camera"
        const val EXTRA_LENS_FACING = "org.genesis.camerabridge.LENS_FACING"
        const val DEFAULT_LENS_FACING = android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
    }

    private val running = AtomicBoolean(false)
    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var sender: Thread? = null
    @Volatile private var serverSocket: ServerSocket? = null
    private val frameLock = Any()
    private var latestRgb: ByteArray? = null
    private var latestWidth: Int = 0
    private var latestHeight: Int = 0
    private var latestSequence: Long = 0
    private var lastCaptureAtNanos: Long = 0
    @Volatile private var requestedLensFacing: Int = DEFAULT_LENS_FACING
    @Volatile private var activeLensFacing: Int = DEFAULT_LENS_FACING

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, notification("Starting Android camera hardware"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val facing = intent?.getIntExtra(EXTRA_LENS_FACING, DEFAULT_LENS_FACING)
            ?: DEFAULT_LENS_FACING
        if (facing != android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT &&
            facing != android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK
        ) {
            reportStatus("Camera unavailable: unsupported lens-facing selection")
            stopSelf()
            return START_NOT_STICKY
        }
        requestedLensFacing = facing
        if (running.compareAndSet(false, true)) {
            reportStatus("Opening ${lensFacingName(requestedLensFacing)} camera; waiting for capture to initialize")
            cameraThread = HandlerThread("GenesisCamera2").also { it.start() }
            cameraHandler = Handler(cameraThread!!.looper)
            sender = thread(name = "GenesisOpticSender", isDaemon = true) { sendFrames() }
            openCamera()
        } else if (activeLensFacing != facing) {
            cameraHandler?.post {
                closeCameraDevice()
                openCamera()
            }
        }
        // Camera access is a while-in-use permission. A sticky background
        // restart cannot reliably reacquire it without a visible activity.
        return START_NOT_STICKY
    }

    private fun openCamera() {
        try {
            val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraIds = manager.cameraIdList
            val cameraId = cameraIds.firstOrNull { id ->
                manager.getCameraCharacteristics(id).get(
                    android.hardware.camera2.CameraCharacteristics.LENS_FACING
                ) == requestedLensFacing
            } ?: run {
                reportStatus("Camera unavailable: Android reports no ${lensFacingName(requestedLensFacing)} camera")
                stopSelf()
                return
            }
            activeLensFacing = requestedLensFacing
            val selectedCamera = lensFacingName(activeLensFacing)
            reportStatus("Opening $selectedCamera camera; configuring Camera2")
            val map = manager.getCameraCharacteristics(cameraId).get(
                android.hardware.camera2.CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
            )
            val size = map?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.filter { it.width in 1..640 && it.height in 1..480 }
                ?.maxByOrNull { it.width * it.height } ?: run {
                    reportStatus("Camera unavailable: no YUV_420_888 size fits the 640x480 optic limit")
                    stopSelf()
                    return
                }
            reader = ImageReader.newInstance(
                size.width,
                size.height,
                ImageFormat.YUV_420_888,
                2
            )
            reader!!.setOnImageAvailableListener({ source ->
                val image: Image = source.acquireLatestImage() ?: return@setOnImageAvailableListener
                val captureAt = System.nanoTime()
                if (lastCaptureAtNanos != 0L &&
                    captureAt - lastCaptureAtNanos < 100_000_000L
                ) {
                    image.close()
                    return@setOnImageAvailableListener
                }
                lastCaptureAtNanos = captureAt
                image.use {
                    try {
                        val rgb = yuv420ToRgb(it)
                        synchronized(frameLock) {
                            latestRgb = rgb
                            latestWidth = it.width
                            latestHeight = it.height
                            latestSequence += 1
                        }
                    } catch (error: Exception) {
                        reportStatus("Camera frame conversion failed: ${error.message ?: error.javaClass.simpleName}")
                        stopSelf()
                    }
                }
            }, cameraHandler)
            manager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) {
                    if (!running.get()) {
                        device.close()
                        return
                    }
                    camera = device
                    reportStatus("$selectedCamera camera opened; configuring capture session")
                    try {
                        device.createCaptureSession(
                            listOf(reader!!.surface),
                            object : CameraCaptureSession.StateCallback() {
                                override fun onConfigured(configured: CameraCaptureSession) {
                                    if (!running.get()) {
                                        configured.close()
                                        return
                                    }
                                    session = configured
                                    try {
                                        val characteristics = manager.getCameraCharacteristics(cameraId)
                                        val requestBuilder = device.createCaptureRequest(
                                            CameraDevice.TEMPLATE_PREVIEW
                                        ).apply {
                                            addTarget(reader!!.surface)
                                            set(
                                                CaptureRequest.CONTROL_MODE,
                                                CaptureRequest.CONTROL_MODE_AUTO
                                            )

                                            // Let the sensor adapt exposure to the scene.
                                            val aeModes = characteristics.get(
                                                android.hardware.camera2.CameraCharacteristics.CONTROL_AE_AVAILABLE_MODES
                                            )
                                            if (aeModes?.contains(CaptureRequest.CONTROL_AE_MODE_ON) == true) {
                                                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                                            }

                                            // Continuous autofocus is preferred for a live, changing
                                            // visual field. Fixed-focus cameras simply omit this control.
                                            val afModes = characteristics.get(
                                                android.hardware.camera2.CameraCharacteristics.CONTROL_AF_AVAILABLE_MODES
                                            )
                                            when {
                                                afModes?.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO) == true ->
                                                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_VIDEO)
                                                afModes?.contains(CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE) == true ->
                                                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                                                afModes?.contains(CaptureRequest.CONTROL_AF_MODE_AUTO) == true ->
                                                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                                            }

                                            // Automatic white balance preserves usable color under
                                            // changing indoor and outdoor illumination.
                                            val awbModes = characteristics.get(
                                                android.hardware.camera2.CameraCharacteristics.CONTROL_AWB_AVAILABLE_MODES
                                            )
                                            if (awbModes?.contains(CaptureRequest.CONTROL_AWB_MODE_AUTO) == true) {
                                                set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                                            }
                                        }
                                        // The eye consumes at most 10 frames/s downstream, but
                                        // letting AE operate at a normal sensor cadence avoids
                                        // forcing long exposure times merely to meet that limit.
                                        val ranges = characteristics.get(
                                            android.hardware.camera2.CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
                                        )
                                        ranges?.filter { it.upper <= 30 && it.upper >= 15 }
                                            ?.maxByOrNull { it.upper }
                                            ?.let {
                                                requestBuilder.set(
                                                    CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                                                    it
                                                )
                                            }
                                        configured.setRepeatingRequest(
                                            requestBuilder.build(),
                                            null,
                                            cameraHandler
                                        )
                                        reportStatus("$selectedCamera eye active: automatic exposure, supported continuous focus and automatic white balance; waiting for Genesis daemon")
                                    } catch (error: Exception) {
                                        reportStatus("Camera capture setup failed: ${error.message ?: error.javaClass.simpleName}")
                                        stopSelf()
                                    }
                                }

                                override fun onConfigureFailed(failed: CameraCaptureSession) {
                                    failed.close()
                                    reportStatus("Camera capture session configuration failed")
                                    stopSelf()
                                }
                            },
                            cameraHandler
                        )
                    } catch (error: Exception) {
                        reportStatus("Camera capture-session creation failed: ${error.message ?: error.javaClass.simpleName}")
                        stopSelf()
                    }
                }

                override fun onDisconnected(device: CameraDevice) {
                    device.close()
                    reportStatus("Camera disconnected by Android")
                    stopSelf()
                }

                override fun onError(device: CameraDevice, error: Int) {
                    device.close()
                    reportStatus("Android Camera2 reported device error $error")
                    stopSelf()
                }
            }, cameraHandler)
        } catch (error: SecurityException) {
            reportStatus("Camera permission denied or revoked: ${error.message ?: "SecurityException"}")
            stopSelf()
        } catch (error: Exception) {
            reportStatus("Camera startup failed: ${error.message ?: error.javaClass.simpleName}")
            stopSelf()
        }
    }

    /**
     * Convert Android's flexible YUV 4:2:0 planes to packed RGB24.
     * Row and pixel strides are honored independently for all planes;
     * this is required because Camera2 does not promise contiguous NV21.
     */
    private fun yuv420ToRgb(image: Image): ByteArray {
        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val output = ByteArray(width * height * 3)
        var out = 0

        for (row in 0 until height) {
            val yRow = row * yPlane.rowStride
            val uRow = (row / 2) * uPlane.rowStride
            val vRow = (row / 2) * vPlane.rowStride
            for (column in 0 until width) {
                val y = (yBuffer.get(yRow + column).toInt() and 0xff) - 16
                val chromaColumn = (column / 2)
                val u = (uBuffer.get(uRow + chromaColumn * uPlane.pixelStride).toInt() and 0xff) - 128
                val v = (vBuffer.get(vRow + chromaColumn * vPlane.pixelStride).toInt() and 0xff) - 128
                val c = maxOf(0, y)
                val red = ((298 * c + 409 * v + 128) shr 8).coerceIn(0, 255)
                val green = ((298 * c - 100 * u - 208 * v + 128) shr 8).coerceIn(0, 255)
                val blue = ((298 * c + 516 * u + 128) shr 8).coerceIn(0, 255)
                output[out++] = red.toByte()
                output[out++] = green.toByte()
                output[out++] = blue.toByte()
            }
        }
        return output
    }

    /**
     * Host the hardware-boundary endpoint on loopback. The daemon
     * connects outward and remains the sole owner of the optic mapping;
     * this avoids adding a second inbound listener to the daemon.
     */
    private fun sendFrames() {
        try {
            ServerSocket().use { server ->
                server.reuseAddress = true
                server.bind(InetSocketAddress(InetAddress.getByName("127.0.0.1"), 8765))
                serverSocket = server
                reportStatus("${lensFacingName(activeLensFacing)} camera adapter listening; waiting for Genesis daemon")
                while (running.get()) {
                    val socket = try {
                        server.accept()
                    } catch (error: Exception) {
                        if (!running.get()) break
                        Log.w(TAG, "Camera transport accept failed; retrying", error)
                        Thread.sleep(250)
                        continue
                    }
                    socket.use { connection ->
                        connection.tcpNoDelay = true
                        reportStatus("Genesis daemon connected; waiting for first ${lensFacingName(activeLensFacing)} camera frame")
                        val output = DataOutputStream(
                            BufferedOutputStream(connection.getOutputStream())
                        )
                        var lastSentSequence = 0L
                        var firstFrameSent = false
                        var noFrameNoticeSent = false
                        val connectedAt = System.nanoTime()
                        while (running.get() && !connection.isClosed) {
                            val snapshot = synchronized(frameLock) {
                                val frame = latestRgb
                                if (frame != null && latestSequence != lastSentSequence) {
                                    Triple(
                                        frame.copyOf(),
                                        latestWidth to latestHeight,
                                        latestSequence
                                    )
                                } else {
                                    null
                                }
                            }
                            if (snapshot != null) {
                                val (frame, dimensions, sequence) = snapshot
                                output.write(byteArrayOf(0x47, 0x4f, 0x50, 0x54))
                                output.writeInt(dimensions.first)
                                output.writeInt(dimensions.second)
                                output.writeInt(frame.size)
                                output.write(frame)
                                output.flush()
                                lastSentSequence = sequence
                                if (!firstFrameSent) {
                                    firstFrameSent = true
                                    reportStatus("${lensFacingName(activeLensFacing)} camera frames streaming to Genesis optic nerve")
                                }
                            } else {
                                if (!noFrameNoticeSent &&
                                    System.nanoTime() - connectedAt > 10_000_000_000L
                                ) {
                                    noFrameNoticeSent = true
                                    reportStatus("Genesis daemon connected, but Camera2 has not delivered a ${lensFacingName(activeLensFacing)} camera frame")
                                }
                                Thread.sleep(20)
                            }
                        }
                    }
                    if (running.get()) {
                        reportStatus("Genesis daemon disconnected; waiting for reconnection")
                    }
                }
            }
        } catch (error: Exception) {
            if (running.get()) {
                reportStatus("Camera transport failed: ${error.message ?: error.javaClass.simpleName}")
                stopSelf()
            }
        } finally {
            serverSocket = null
        }
    }

    private fun closeCameraDevice() {
        try {
            session?.stopRepeating()
        } catch (_: Exception) {
        }
        session?.close()
        session = null
        camera?.close()
        camera = null
        reader?.close()
        reader = null
    }

    private fun lensFacingName(facing: Int): String =
        if (facing == android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT) {
            "front"
        } else {
            "rear"
        }

    override fun onDestroy() {
        running.set(false)
        try { serverSocket?.close() } catch (_: Exception) { }
        closeCameraDevice()
        cameraThread?.quitSafely()
        cameraThread = null
        cameraHandler = null
        Log.i(TAG, "Camera service stopped")
        stopForeground(STOP_FOREGROUND_REMOVE)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun reportStatus(message: String) {
        Log.i(TAG, message)
        val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        manager.notify(NOTIFICATION_ID, notification(message))
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                NOTIFICATION_CHANNEL,
                "Genesis camera",
                NotificationManager.IMPORTANCE_LOW
            )
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(channel)
        }
    }

    private fun notification(message: String): Notification {
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, NOTIFICATION_CHANNEL)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }
        return builder.setContentTitle("Genesis optic input")
            .setContentText(message)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .build()
    }
}
