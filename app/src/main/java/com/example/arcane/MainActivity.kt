package com.example.arcane

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.ViewGroup
import android.widget.*
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : Activity() {
    private lateinit var image: ImageView
    private lateinit var status: TextView
    private lateinit var seek: SeekBar
    private lateinit var saveBtn: Button
    private var predictor: Interpreter? = null
    private var transfer: Interpreter? = null
    private var bottleneck: ByteBuffer? = null
    private var content: Bitmap? = null
    private var stylized: Bitmap? = null
    private var result: Bitmap? = null
    private val SIZE = 384
    private val STYLE = 256

    override fun onCreate(b: Bundle?) {
        super.onCreate(b)
        val root = LinearLayout(this)
        root.orientation = LinearLayout.VERTICAL
        root.setPadding(32, 32, 32, 32)
        root.fitsSystemWindows = true
        val title = TextView(this)
        title.text = "Arcane Photo"
        title.textSize = 26f
        val pick = Button(this)
        pick.text = "Choose photo"
        image = ImageView(this)
        image.scaleType = ImageView.ScaleType.FIT_CENTER
        val label = TextView(this)
        label.text = "Style strength"
        seek = SeekBar(this)
        seek.max = 100
        seek.progress = 100
        saveBtn = Button(this)
        saveBtn.text = "Save to gallery"
        saveBtn.isEnabled = false
        status = TextView(this)
        status.text = "Loading AI model..."
        root.addView(title)
        root.addView(pick)
        root.addView(image, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        root.addView(label)
        root.addView(seek)
        root.addView(saveBtn)
        root.addView(status)
        setContentView(root)

        pick.setOnClickListener {
            startActivityForResult(Intent(Intent.ACTION_GET_CONTENT).setType("image/*"), 1)
        }
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, u: Boolean) { blend() }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })
        saveBtn.setOnClickListener { save() }
        Thread { loadModels() }.start()
    }

    private fun asset(name: String): ByteBuffer {
        val bytes = assets.open(name).use { it.readBytes() }
        val bb = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        bb.put(bytes)
        bb.rewind()
        return bb
    }

    private fun square(src: Bitmap, size: Int): Bitmap {
        val s = minOf(src.width, src.height)
        val x = (src.width - s) / 2
        val y = (src.height - s) / 2
        val crop = Bitmap.createBitmap(src, x, y, s, s)
        return Bitmap.createScaledBitmap(crop, size, size, true)
    }

    private fun toBuffer(bmp: Bitmap, size: Int): ByteBuffer {
        val bb = ByteBuffer.allocateDirect(size * size * 3 * 4).order(ByteOrder.nativeOrder())
        val px = IntArray(size * size)
        bmp.getPixels(px, 0, size, 0, 0, size, size)
        for (p in px) {
            bb.putFloat(((p shr 16) and 255) / 255f)
            bb.putFloat(((p shr 8) and 255) / 255f)
            bb.putFloat((p and 255) / 255f)
        }
        bb.rewind()
        return bb
    }

    private fun loadModels() {
        try {
            val pr = Interpreter(asset("style_predict.tflite"))
            val tr = Interpreter(asset("style_transfer.tflite"))
            val opts = BitmapFactory.Options()
            opts.inSampleSize = 2
            val style = assets.open("style.jpg").use { BitmapFactory.decodeStream(it, null, opts) }!!
            val input = toBuffer(square(style, STYLE), STYLE)
            val out = ByteBuffer.allocateDirect(100 * 4).order(ByteOrder.nativeOrder())
            pr.run(input, out)
            out.rewind()
            predictor = pr
            transfer = tr
            bottleneck = out
            runOnUiThread { status.text = "Ready. Choose a photo." }
        } catch (e: Exception) {
            runOnUiThread { status.text = "Error: " + e.message }
        }
    }

    override fun onActivityResult(req: Int, res: Int, data: Intent?) {
        super.onActivityResult(req, res, data)
        if (req != 1 || res != RESULT_OK) return
        val uri: Uri = data?.data ?: return
        if (bottleneck == null) {
            status.text = "Model still loading, wait a moment."
            return
        }
        status.text = "Painting... please wait"
        saveBtn.isEnabled = false
        Thread { process(uri) }.start()
    }

    private fun process(uri: Uri) {
        try {
            val opts = BitmapFactory.Options()
            opts.inSampleSize = 2
            val raw = contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it, null, opts) }!!
            val exif = contentResolver.openInputStream(uri)!!.use { android.media.ExifInterface(it) }
            val deg = when (exif.getAttributeInt(android.media.ExifInterface.TAG_ORIENTATION, 1)) {
                6 -> 90f
                3 -> 180f
                8 -> 270f
                else -> 0f
            }
            val m = Matrix()
            m.postRotate(deg)
            val rot = Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
            val sq = square(rot, SIZE)
            val inBuf = toBuffer(sq, SIZE)
            val bn = bottleneck!!
            bn.rewind()
            val t = transfer!!
            val a: Any = if (t.getInputTensor(0).shape().last() == 100) bn else inBuf
            val b: Any = if (t.getInputTensor(1).shape().last() == 100) bn else inBuf
            val outBuf = ByteBuffer.allocateDirect(SIZE * SIZE * 3 * 4).order(ByteOrder.nativeOrder())
            t.runForMultipleInputsOutputs(arrayOf(a, b), mapOf(0 to outBuf))
            outBuf.rewind()
            val px = IntArray(SIZE * SIZE)
            for (i in px.indices) {
                val r = (outBuf.getFloat().coerceIn(0f, 1f) * 255).toInt()
                val g = (outBuf.getFloat().coerceIn(0f, 1f) * 255).toInt()
                val bl = (outBuf.getFloat().coerceIn(0f, 1f) * 255).toInt()
                px[i] = Color.rgb(r, g, bl)
            }
            content = sq
            stylized = Bitmap.createBitmap(px, SIZE, SIZE, Bitmap.Config.ARGB_8888)
            runOnUiThread {
                blend()
                saveBtn.isEnabled = true
                status.text = "Done. Move the slider to change strength."
            }
        } catch (e: Exception) {
            runOnUiThread { status.text = "Error: " + e.message }
        }
    }

    private fun blend() {
        val c = content ?: return
        val s = stylized ?: return
        val out = c.copy(Bitmap.Config.ARGB_8888, true)
        val p = Paint()
        p.alpha = seek.progress * 255 / 100
        Canvas(out).drawBitmap(s, 0f, 0f, p)
        result = out
        image.setImageBitmap(out)
    }

    private fun save() {
        val r = result ?: return
        val v = ContentValues()
        v.put(MediaStore.Images.Media.DISPLAY_NAME, "arcane_" + System.currentTimeMillis() + ".jpg")
        v.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        v.put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ArcanePhoto")
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, v)
        if (uri == null) {
            status.text = "Could not save."
            return
        }
        contentResolver.openOutputStream(uri)!!.use { r.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        status.text = "Saved to Pictures/ArcanePhoto"
    }
}
