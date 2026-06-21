// kotlin_demo.kt
package com.nexus.eval

object AppDatabase {
    val databaseName = "nexus_eval_db"
}

class UserRepository {
    fun fetchUser(userId: String): String {
        return "User-$userId"
    }
}
