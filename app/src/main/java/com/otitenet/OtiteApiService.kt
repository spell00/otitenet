package com.otitenet

import com.google.gson.annotations.SerializedName
import okhttp3.MultipartBody
import okhttp3.RequestBody
import retrofit2.http.*

data class AnalysisResult(
    val prediction: String,
    val confidence: Double,
    val filename: String,
    val timestamp: String? = null
)

data class DeploymentManifest(
    val labels: List<String>,
    @SerializedName("input") val input: ModelInput,
    val files: Map<String, String>
)

data class ModelInput(
    @SerializedName("image_size") val imageSize: List<Int>
)

interface OtiteApiService {
    @GET("deployment/current")
    suspend fun getManifest(): DeploymentManifest

    @Multipart
    @POST("analyze")
    suspend fun analyzeImage(
        @Part("person_id") personId: RequestBody,
        @Part file: MultipartBody.Part
    ): AnalysisResult

    @GET("results/{person_id}")
    suspend fun getHistory(@Path("person_id") personId: Int): List<AnalysisResult>
}
