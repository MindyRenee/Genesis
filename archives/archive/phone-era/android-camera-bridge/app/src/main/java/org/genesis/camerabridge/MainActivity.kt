package org.genesis.camerabridge

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

class MainActivity : Activity() {
    private companion object {
        const val EXTRA_STOP_CAMERA = "org.genesis.camerabridge.STOP_CAMERA"
        const val EXTRA_LENS_FACING = "org.genesis.camerabridge.LENS_FACING"
        const val FRONT_CAMERA = android.hardware.camera2.CameraCharacteristics.LENS_FACING_FRONT
        const val REAR_CAMERA = android.hardware.camera2.CameraCharacteristics.LENS_FACING_BACK
    }

    private lateinit var status: TextView
    private var selectedLensFacing: Int = FRONT_CAMERA

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply {
            text = "Genesis Camera2 input publishes frames to the native optic nerve in the Termux daemon."
            textSize = 16f
            setPadding(24, 24, 24, 24)
        }
        val front = Button(this).apply {
            text = "Use front camera"
            setOnClickListener { requestCameraAndStart(FRONT_CAMERA) }
        }
        val rear = Button(this).apply {
            text = "Use rear camera"
            setOnClickListener { requestCameraAndStart(REAR_CAMERA) }
        }
        val start = Button(this).apply {
            text = "Start / retry selected camera"
            setOnClickListener { requestCameraAndStart(selectedLensFacing) }
        }
        val mic = Button(this).apply {
            text = "Start microphone"
            setOnClickListener { requestMicAndStart() }
        }
        val stop = Button(this).apply {
            text = "Stop streams"
            setOnClickListener {
                stopService(Intent(this@MainActivity, CameraStreamService::class.java))
                stopService(Intent(this@MainActivity, AudioStreamService::class.java))
                status.text = "Streams stopped."
            }
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(status)
            addView(front)
            addView(rear)
            addView(start)
            addView(mic)
            addView(stop)
        })
        // The Genesis stop command enters through this explicit activity
        // action so the service is stopped inside its owning application.
        if (intent.getBooleanExtra(EXTRA_STOP_CAMERA, false)) {
            stopService(Intent(this, CameraStreamService::class.java))
            stopService(Intent(this, AudioStreamService::class.java))
            status.text = "Streams stopped."
            finish()
            return
        }

        // Genesis launches this activity from its supported Termux entry
        // point. Request permission here while the activity is visible;
        // Android will not grant a camera foreground service access from
        // an ordinary background start.
        if (savedInstanceState == null) {
            requestCameraAndStart()
        }
    }

    private fun requestCameraAndStart(lensFacing: Int = FRONT_CAMERA) {
        selectedLensFacing = lensFacing
        val cameraName = if (lensFacing == FRONT_CAMERA) "front" else "rear"
        status.text = "Selected $cameraName camera."
        val missing = mutableListOf<String>()
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            missing += Manifest.permission.CAMERA
        }
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            missing += Manifest.permission.RECORD_AUDIO
        }
        if (missing.isNotEmpty()) {
            requestPermissions(missing.toTypedArray(), 41)
            return
        }
        startStreamService()
        startAudioService()
    }

    private fun requestMicAndStart() {
        status.text = "Selected microphone."
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 42)
            return
        }
        startAudioService()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 41 && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startStreamService()
            if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                startAudioService()
            }
        } else if (requestCode == 42 && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startAudioService()
        } else {
            status.text = "Camera and microphone permissions are required to stream senses."
        }
    }

    private fun startStreamService() {
        try {
            startForegroundService(
                Intent(this, CameraStreamService::class.java)
                    .putExtra(EXTRA_LENS_FACING, selectedLensFacing)
            )
            val cameraName = if (selectedLensFacing == FRONT_CAMERA) "front" else "rear"
            status.text = "Starting $cameraName camera. The notification reports camera and daemon connection state."
        } catch (error: SecurityException) {
            status.text = "Android refused camera startup: ${error.message ?: "permission or foreground-service restriction"}"
        } catch (error: Exception) {
            status.text = "Could not start camera service: ${error.message ?: error.javaClass.simpleName}"
        }
    }

    private fun startAudioService() {
        try {
            startForegroundService(Intent(this, AudioStreamService::class.java))
            status.text = "Starting microphone. The notification reports audio and daemon connection state."
        } catch (error: SecurityException) {
            status.text = "Android refused microphone startup: ${error.message ?: "permission or foreground-service restriction"}"
        } catch (error: Exception) {
            status.text = "Could not start microphone service: ${error.message ?: error.javaClass.simpleName}"
        }
    }
}
