package com.otitenet

import android.content.Context
import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "local_results")
data class LocalResult(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val prediction: String,
    val confidence: Double,
    val timestamp: Long = System.currentTimeMillis(),
    val imagePath: String? = null
)

@Dao
interface ResultDao {
    @Query("SELECT * FROM local_results ORDER BY timestamp DESC")
    fun getAllResults(): Flow<List<LocalResult>>

    @Insert
    suspend fun insert(result: LocalResult)
}

@Database(entities = [LocalResult::class], version = 1)
abstract class AppDatabase : RoomDatabase() {
    abstract fun resultDao(): ResultDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "otitenet_database"
                ).build()
                INSTANCE = instance
                instance
            }
        }
    }
}
