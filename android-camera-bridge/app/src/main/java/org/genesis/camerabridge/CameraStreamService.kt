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
import android.util.Size
import java.io.BufferedOutputStream
import java.io.DataOutputStream
import java.net.InetSocketAddress
import java.net.Socket
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

class CameraStreamService : Service() {
    private val running = AtomicBoolean(false)
    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var cameraThread: HandlerThread? = null
    private var cameraHandler: Handler? = null
    private var sender: Thread? = null
    private val frameLock = Any()
    private var latestJpeg: ByteArray? = null
    private var latestSequence: Long = 0

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(7, notification("Camera stream active"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (running.compareAndSet(false, true)) {
            cameraThread = HandlerThread("GenesisCamera2").also { it.start() }
            cameraHandler = Handler(cameraThread!!.looper)
            openCamera()
            sender = thread(name = "GenesisFrameSender", isDaemon = true) { sendFrames() }
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
            val size = map?.getOutputSizes(ImageFormat.JPEG)
                ?.filter { it.width <= 640 && it.height <= 480 }
                ?.maxByOrNull { it.width * it.height } ?: Size(640, 480)
            reader = ImageReader.newInstance(size.width, size.height, ImageFormat.JPEG, 2)
            reader!!.setOnImageAvailableListener({ source ->
                val image: Image = source.acquireLatestImage() ?: return@setOnImageAvailableListener
                image.use {
                    val buffer = it.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    synchronized(frameLock) {
                        latestJpeg = bytes
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
                                    val requestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                                        addTarget(reader!!.surface)
                                        set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO)
                                    }
                                    val ranges = manager.getCameraCharacteristics(cameraId).get(
                                        android.hardware.camera2.CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES
                                    )
                                    ranges?.filter { it.upper <= 10 && it.lower <= 5 }
                                        ?.maxByOrNull { it.upper }
                                        ?.let { requestBuilder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it) }
                                    val request = requestBuilder.build()
                                    configured.setRepeatingRequest(request, null, cameraHandler)
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

    private fun sendFrames() {
        var lastSentSequence = 0L
        while (running.get()) {
            try {
                Socket().use { socket ->
                    socket.connect(InetSocketAddress("127.0.0.1", 8765), 1000)
                    socket.tcpNoDelay = true
                    DataOutputStream(BufferedOutputStream(socket.getOutputStream())).use { output ->
                        while (running.get()) {
                            val snapshot = synchronized(frameLock) {
                                if (latestJpeg != null && latestSequence != lastSentSequence) {
                                    Pair(latestJpeg!!.copyOf(), latestSequence)
                                } else null
                            }
                            if (snapshot != null) {
                                val (frame, sequence) = snapshot
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
            } catch (_: Exception) {
                try { Thread.sleep(1000) } catch (_: InterruptedException) { return }
            }
        }
    }

    override fun onDestroy() {
        running.set(false)
        try { session?.stopRepeating() } catch (_: Exception) { }
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
        return builder.setContentTitle("Genesis Camera Bridge")
            .setContentText(message)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setOngoing(true)
            .build()
    }
}
