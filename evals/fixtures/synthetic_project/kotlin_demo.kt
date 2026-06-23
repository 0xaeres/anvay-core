// kotlin_demo.kt
package com.anvay.eval

object AppDatabase {
    val databaseName = "anvay_eval_db"
}

class UserRepository {
    fun fetchUser(userId: String): String {
        return "User-$userId"
    }
}
