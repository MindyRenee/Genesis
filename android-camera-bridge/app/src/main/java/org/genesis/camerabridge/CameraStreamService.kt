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

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(7, notification("Camera input connected to Genesis optic nerve"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (running.compareAndSet(false, true)) {
            cameraThread = HandlerThread("GenesisCamera2").also { it.start() }
            cameraHandler = Handler(cameraThread!!.looper)
            openCamera()
            sender = thread(name = "GenesisOpticSender", isDaemon = true) { sendFrames() }
        }
        return START_STICKY
    }

    private fun openCamera() {
        try {
            val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            val cameraId = manager.cameraIdList.firstOrNull { id ->
                manager.getCameraCharacteristics(id).get(
                    android.hardware.camera2.CameraCharacteristics.LENS_FACING
                ) == android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK
            } ?: manager.cameraIdList.first()
            val map = manager.getCameraCharacteristics(cameraId).get(
                android.hardware.camera2.CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
            )
            val size = map?.getOutputSizes(ImageFormat.YUV_420_888)
                ?.filter { it.width in 1..640 && it.height in 1..480 }
                ?.maxByOrNull { it.width * it.height } ?: run {
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
                    val rgb = yuv420ToRgb(it)
                    synchronized(frameLock) {
                        latestRgb = rgb
                        latestWidth = it.width
                        latestHeight = it.height
                        latestSequence += 1
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
                                    val requestBuilder = device.createCaptureRequest(
                                        CameraDevice.TEMPLATE_PREVIEW
                                    ).apply {
                                        addTarget(reader!!.surface)
                                        set(
                                            CaptureRequest.CONTROL_MODE,
                                            CaptureRequest.CONTROL_MODE_AUTO
                                        )
                                    }
                                    val ranges = manager.getCameraCharacteristics(cameraId).get(
                                        android.hardware.camera2.CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
                                    )
                                    ranges?.filter { it.upper <= 10 && it.lower <= 5 }
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
                                }

                                override fun onConfigureFailed(failed: CameraCaptureSession) {
                                    failed.close()
                                    stopSelf()
                                }
                            },
                            cameraHandler
                        )
                    } catch (_: Exception) {
                        stopSelf()
                    }
                }

                override fun onDisconnected(device: CameraDevice) {
                    device.close()
                    stopSelf()
                }

                override fun onError(device: CameraDevice, error: Int) {
                    device.close()
                    stopSelf()
                }
            }, cameraHandler)
        } catch (_: SecurityException) {
            stopSelf()
        } catch (_: Exception) {
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
                server.bind(InetSocketAddress(InetAddress.getLoopbackAddress(), 8765))
                serverSocket = server
                while (running.get()) {
                    val socket = try {
                        server.accept()
                    } catch (_: Exception) {
                        if (!running.get()) break
                        continue
                    }
                    socket.use { connection ->
                        connection.tcpNoDelay = true
                        val output = DataOutputStream(
                            BufferedOutputStream(connection.getOutputStream())
                        )
                        var lastSentSequence = 0L
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
                            } else {
                                Thread.sleep(20)
                            }
                        }
                    }
                }
            }
        } catch (_: Exception) {
            if (running.get()) stopSelf()
        } finally {
            serverSocket = null
        }
    }

    override fun onDestroy() {
        running.set(false)
        try { serverSocket?.close() } catch (_: Exception) { }
        try {
            session?.stopRepeating()
        } catch (_: Exception) {
        }
        session?.close()
        camera?.close()
        reader?.close()
        cameraThread?.quitSafely()
        cameraThread = null
        cameraHandler = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "genesis_camera",
                "Genesis camera",
                NotificationManager.IMPORTANCE_LOW
            )
            (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager)
                .createNotificationChannel(channel)
        }
    }

    private fun notification(message: String): Notification {
        val builder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, "genesis_camera")
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
