package com.otitenet

import android.content.Context
import android.graphics.Bitmap
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File

class InferenceManager(private val context: Context) {
    private var module: Module? = null
    private var labels: List<String> = listOf("Normal", "Not Normal")
    private var inputSize: Int = 224

    fun loadModel(modelFile: File, labelList: List<String>, size: Int = 224) {
        module = LiteModuleLoader.load(modelFile.absolutePath)
        labels = labelList
        inputSize = size
    }

    fun isModelLoaded(): Boolean = module != null

    fun analyze(bitmap: Bitmap): AnalysisResult {
        if (module == null) {
            throw IllegalStateException("Model not loaded")
        }

        // Resize bitmap to model input size
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        // Preprocess (normalize mean/std usually 0.485, 0.456, 0.406 / 0.229, 0.224, 0.225)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            resizedBitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
            TensorImageUtils.TORCHVISION_NORM_STD_RGB
        )

        // Inference
        val outputTensor = module!!.forward(IValue.from(inputTensor)).toTensor()
        val scores = outputTensor.dataAsFloatArray

        // Post-process: Find max index (assuming Softmax output)
        var maxIdx = 0
        var maxScore = -1f
        for (i in scores.indices) {
            if (scores[i] > maxScore) {
                maxScore = scores[i]
                maxIdx = i
            }
        }

        return AnalysisResult(
            prediction = labels.getOrElse(maxIdx) { "Unknown" },
            confidence = maxScore.toDouble(),
            filename = "local_inference_${System.currentTimeMillis()}.jpg"
        )
    }
}
