package org.genesis.camerabridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
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
 * Android's hardware-facing microphone adapter.
 *
 * It captures AudioRecord PCM (16 kHz mono s16 — the rate her Vosk
 * ear is tuned for) and streams it raw to the Genesis side over
 * loopback, mirroring the optic service: the body side connects
 * outward and owns consumption; this adapter performs no recognition,
 * filtering, or perception.
 */
class AudioStreamService : Service() {
    private companion object {
        const val TAG = "GenesisAudio"
        const val NOTIFICATION_ID = 8
        const val NOTIFICATION_CHANNEL = "genesis_audio"
        const val SAMPLE_RATE = 16000
        const val PORT = 8766
        const val CHUNK = 2048
    }

    private val running = AtomicBoolean(false)
    private var recorder: AudioRecord? = null
    private var serverSocket: ServerSocket? = null
    private var sender: Thread? = null
    private var captureThread: Thread? = null
    private val bufferLock = Any()
    private var pending = ByteArray(0)

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, notification("Starting Android microphone hardware"))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (running.compareAndSet(false, true)) {
            reportStatus("Opening microphone; waiting for capture to initialize")
            sender = thread(name = "GenesisAudioSender", isDaemon = true) { serve() }
            captureThread = thread(name = "GenesisAudioCapture", isDaemon = true) { capture() }
        }
        // Mic access is a while-in-use permission. A sticky background
        // restart cannot reliably reacquire it without a visible activity.
        return START_NOT_STICKY
    }

    private fun capture() {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        try {
            recorder = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuf * 2, CHUNK * 4)
            )
        } catch (error: SecurityException) {
            reportStatus("Microphone permission denied or revoked")
            stopSelf()
            return
        }
        val rec = recorder ?: return
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            reportStatus("Microphone unavailable: AudioRecord failed to initialize")
            stopSelf()
            return
        }
        reportStatus("Microphone active at ${SAMPLE_RATE}Hz mono s16; waiting for Genesis")
        rec.startRecording()
        val chunk = ByteArray(CHUNK)
        while (running.get()) {
            val n = rec.read(chunk, 0, chunk.size)
            if (n > 0) {
                synchronized(bufferLock) {
                    pending += chunk.copyOf(n)
                    // Bound the queue: a disconnected consumer must not
                    // let capture memory grow without limit.
                    if (pending.size > SAMPLE_RATE * 4) {
                        pending = pending.copyOfRange(pending.size - SAMPLE_RATE * 2, pending.size)
                    }
                }
            } else {
                Thread.sleep(10)
            }
        }
    }

    /**
     * Host the audio endpoint on loopback. The consumer connects
     * outward and owns the auditory mapping; a single 16-byte header
     * ("GAUD", rate, channels, bytes/sample) prefixes the raw PCM.
     */
    private fun serve() {
        try {
            ServerSocket().use { server ->
                server.reuseAddress = true
                server.bind(InetSocketAddress(InetAddress.getByName("127.0.0.1"), PORT))
                serverSocket = server
                reportStatus("Microphone adapter listening on 127.0.0.1:$PORT")
                while (running.get()) {
                    val socket = try {
                        server.accept()
                    } catch (error: Exception) {
                        if (!running.get()) break
                        Log.w(TAG, "Audio transport accept failed; retrying", error)
                        Thread.sleep(250)
                        continue
                    }
                    socket.use { connection ->
                        connection.tcpNoDelay = true
                        reportStatus("Genesis connected; streaming microphone PCM")
                        val output = DataOutputStream(
                            BufferedOutputStream(connection.getOutputStream())
                        )
                        output.write(byteArrayOf(0x47, 0x41, 0x55, 0x44))
                        output.writeInt(SAMPLE_RATE)
                        output.writeInt(1)
                        output.writeInt(2)
                        output.flush()
                        synchronized(bufferLock) { pending = ByteArray(0) }
                        while (running.get() && !connection.isClosed) {
                            val frame = synchronized(bufferLock) {
                                if (pending.isNotEmpty()) {
                                    val out = pending
                                    pending = ByteArray(0)
                                    out
                                } else {
                                    null
                                }
                            }
                            if (frame != null) {
                                output.write(frame)
                                output.flush()
                            } else {
                                Thread.sleep(10)
                            }
                        }
                    }
                    if (running.get()) {
                        reportStatus("Genesis disconnected; waiting for reconnection")
                    }
                }
            }
        } catch (error: Exception) {
            if (running.get()) {
                reportStatus("Microphone transport failed: ${error.message ?: error.javaClass.simpleName}")
                stopSelf()
            }
        } finally {
            serverSocket = null
        }
    }

    override fun onDestroy() {
        running.set(false)
        try { serverSocket?.close() } catch (_: Exception) { }
        try {
            recorder?.stop()
        } catch (_: Exception) {
        }
        recorder?.release()
        recorder = null
        Log.i(TAG, "Audio service stopped")
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
                "Genesis microphone",
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
        return builder.setContentTitle("Genesis auditory input")
            .setContentText(message)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .build()
    }
}
