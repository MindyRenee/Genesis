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
    private lateinit var status: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        status = TextView(this).apply {
            text = "Genesis Camera2 input publishes frames to the native optic nerve in the Termux daemon."
            textSize = 16f
            setPadding(24, 24, 24, 24)
        }
        val start = Button(this).apply {
            text = "Start camera stream"
            setOnClickListener { requestCameraAndStart() }
        }
        val stop = Button(this).apply {
            text = "Stop camera stream"
            setOnClickListener {
                stopService(Intent(this@MainActivity, CameraStreamService::class.java))
                status.text = "Camera stream stopped."
            }
        }
        setContentView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(status)
            addView(start)
            addView(stop)
        })
    }

    private fun requestCameraAndStart() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 41)
            return
        }
        startStreamService()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 41 && grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) {
            startStreamService()
        } else if (requestCode == 41) {
            status.text = "Camera permission is required to stream frames."
        }
    }

    private fun startStreamService() {
        startForegroundService(Intent(this, CameraStreamService::class.java))
        status.text = "Connecting Camera2 to the Genesis optic nerve. Keep the daemon running in Termux."
    }
}
