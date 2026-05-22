package com.otitenet

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class LocalDatabaseTest {
    private lateinit var db: AppDatabase
    private lateinit var dao: ResultDao

    @Before
    fun createDb() {
        db = Room.inMemoryDatabaseBuilder(
            ApplicationProvider.getApplicationContext(),
            AppDatabase::class.java
        ).build()
        dao = db.resultDao()
    }

    @After
    fun closeDb() {
        db.close()
    }

    @Test
    fun writeAndReadResult() = runBlocking {
        val result = LocalResult(prediction = "Normal", confidence = 0.99)
        dao.insert(result)
        val allResults = dao.getAllResults().first()
        assertEquals(allResults[0].prediction, "Normal")
        assertEquals(allResults[0].confidence, 0.99, 0.001)
    }
}
