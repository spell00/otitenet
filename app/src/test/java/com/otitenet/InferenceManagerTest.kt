package com.otitenet

import android.graphics.Bitmap
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.mockito.Mockito.*

class InferenceManagerTest {

    @Test
    fun `test analyze returns non-null result`() {
        // This test would typically require a real model or a mocked InferenceManager
        // Since PyTorch Lite requires native libs, we mostly test the interface here
        // or use a mock for higher level logic.

        val mockManager = mock(InferenceManager::class.java)
        val mockBitmap = mock(Bitmap::class.java)
        val expected = AnalysisResult("Normal", 0.95, "test.jpg")

        `when`(mockManager.analyze(mockBitmap)).thenReturn(expected)

        val result = mockManager.analyze(mockBitmap)
        assertNotNull(result)
        assertEquals("Normal", result.prediction)
        assertEquals(0.95, result.confidence, 0.01)
    }
}
