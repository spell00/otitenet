package com.otitenet

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.PhotoCamera
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody

private const val BASE_URL = "http://10.0.2.2:8000/"

val retrofit: Retrofit = Retrofit.Builder()
    .baseUrl(BASE_URL)
    .addConverterFactory(GsonConverterFactory.create())
    .build()

val apiService: OtiteApiService = retrofit.create(OtiteApiService::class.java)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(color = MaterialTheme.colorScheme.background) {
                    OtiteApp()
                }
            }
        }
    }
}

@Composable
fun OtiteApp() {
    var selectedTab by remember { mutableIntStateOf(0) }
    val context = LocalContext.current
    val database = remember { AppDatabase.getDatabase(context) }
    val inferenceManager = remember { InferenceManager(context) }

    Scaffold(
        bottomBar = {
            NavigationBar {
                NavigationBarItem(
                    selected = selectedTab == 0,
                    onClick = { selectedTab = 0 },
                    icon = { Icon(Icons.Default.PhotoCamera, contentDescription = "Analyze") },
                    label = { Text("Analyze") }
                )
                NavigationBarItem(
                    selected = selectedTab == 1,
                    onClick = { selectedTab = 1 },
                    icon = { Icon(Icons.Default.History, contentDescription = "History") },
                    label = { Text("History") }
                )
            }
        }
    ) { padding ->
        Box(modifier = Modifier.padding(padding)) {
            when (selectedTab) {
                0 -> AnalysisScreen(inferenceManager, database)
                1 -> HistoryScreen(database)
            }
        }
    }
}

@Composable
fun AnalysisScreen(inferenceManager: InferenceManager, database: AppDatabase) {
    var imageUri by remember { mutableStateOf<Uri?>(null) }
    var result by remember { mutableStateOf<AnalysisResult?>(null) }
    var statusMessage by remember { mutableStateOf("Ready") }
    var isProcessing by remember { mutableStateOf(false) }
    var isDownloading by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val context = LocalContext.current

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? -> imageUri = uri }

    LaunchedEffect(Unit) {
        withContext(Dispatchers.IO) {
            try {
                statusMessage = "Checking for updates..."
                val manifest = apiService.getManifest()
                val modelFileName = manifest.files["model"] ?: "model.ptl"
                val modelFile = File(context.filesDir, modelFileName)

                if (!modelFile.exists()) {
                    isDownloading = true
                    statusMessage = "Downloading model..."
                    val url = URL("${BASE_URL}deployment/current/files/$modelFileName")
                    url.openStream().use { input ->
                        FileOutputStream(modelFile).use { output ->
                            input.copyTo(output)
                        }
                    }
                }

                inferenceManager.loadModel(
                    modelFile,
                    manifest.labels,
                    manifest.input.imageSize.getOrElse(0) { 224 }
                )
                statusMessage = "System Ready: ${manifest.labels.joinToString()}"
                isDownloading = false
            } catch (e: Exception) {
                statusMessage = "Offline Mode (Using cached model)"
                val modelFile = File(context.filesDir, "model.ptl")
                if (modelFile.exists()) {
                    inferenceManager.loadModel(modelFile, listOf("Normal", "Abnormal"))
                }
                isDownloading = false
            }
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(statusMessage, style = MaterialTheme.typography.labelSmall)

        if (isDownloading || isProcessing) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        Card(modifier = Modifier.size(280.dp).padding(8.dp)) {
            if (imageUri != null) {
                AsyncImage(
                    model = imageUri,
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
            } else {
                Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("No Image Selected", style = MaterialTheme.typography.bodyMedium)
                }
            }
        }

        Spacer(Modifier.height(16.dp))

        Button(
            onClick = { launcher.launch("image/*") },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("1. Select Photo")
        }

        Spacer(Modifier.height(8.dp))

        // This is your "Submit" button for Offline Analysis
        Button(
            onClick = {
                isProcessing = true
                scope.launch {
                    val bitmap = withContext(Dispatchers.IO) {
                        context.contentResolver.openInputStream(imageUri!!)?.use {
                            BitmapFactory.decodeStream(it)
                        }
                    }
                    if (bitmap != null) {
                        val analysis = inferenceManager.analyze(bitmap)
                        result = analysis
                        database.resultDao().insert(
                            LocalResult(prediction = analysis.prediction, confidence = analysis.confidence)
                        )
                    }
                    isProcessing = false
                }
            },
            enabled = imageUri != null && inferenceManager.isModelLoaded() && !isProcessing,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(if (isProcessing) "Analyzing..." else "2. Run Local Analysis (Offline)")
        }

        result?.let {
            Spacer(Modifier.height(16.dp))
            Text("Result: ${it.prediction}", style = MaterialTheme.typography.headlineMedium)
            Text("Confidence: ${(it.confidence * 100).toInt()}%")
        }
    }
}

@Composable
fun HistoryScreen(database: AppDatabase) {
    val results by database.resultDao().getAllResults().collectAsState(initial = emptyList())

    LazyColumn(modifier = Modifier.fillMaxSize()) {
        items(results) { item ->
            ListItem(
                headlineContent = { Text(item.prediction) },
                supportingContent = { Text("Confidence: ${(item.confidence * 100).toInt()}%") },
                trailingContent = {
                    val date = java.text.SimpleDateFormat("dd/MM/yy HH:mm", java.util.Locale.getDefault())
                        .format(java.util.Date(item.timestamp))
                    Text(date)
                }
            )
            Divider()
        }
    }
}
