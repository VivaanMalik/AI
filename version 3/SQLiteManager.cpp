#include "header.hpp"

sqlite3* setup_sqlite3_db() {
    sqlite3* db_pointer; 
    int response;
    response = sqlite3_open(":memory:", &db_pointer); // open in memory
    if (response) {
        cerr << "Can't open in-memory database: " << sqlite3_errmsg(db_pointer) << endl; ;
    } else {
        cout << "In-memory database opened successfully." << endl; 
    }

    
    const char* sql = "CREATE TABLE gpu_work_queue (id INTEGER, description TEXT, created_on FLOAT);"; 
    char* errMsg = nullptr; 
    response = sqlite3_exec(db_pointer, sql, nullptr, nullptr, &errMsg);
    if (response != SQLITE_OK) {
        cerr << "SQL error: " << errMsg << endl; 
        sqlite3_free(errMsg);
    } else {
        cout << "Table created successfully.\n"; 
    }

    return db_pointer;
}

bool write_to_sqlite3_db(sqlite3* db_pointer, string values) {
    const string sql = "INSERT INTO gpu_work_queue VALUES (" + values + ");"; 
    char* errMsg = nullptr;

    int response = sqlite3_exec(db_pointer, sql.c_str(), nullptr, nullptr, &errMsg);

    if (response != SQLITE_OK) {
        cerr << "Insert failed: " << errMsg << endl;
        sqlite3_free(errMsg);
        cout << sql << endl << endl;
        return false;
    }

    cout << "Inserted value: " + values + "\n";
    return true;
}

void close_sqlite3_db(sqlite3* db_pointer) {
    sqlite3_close(db_pointer);
    cout << "Database closed successfully.\n"; 
}

optional<json> read_from_sqlite3_db(sqlite3* db_pointer, int id_to_confirm_with) {
    const char* sql = "SELECT id, description, created_on FROM gpu_work_queue ORDER BY created_on ASC LIMIT 1;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db_pointer, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        cerr << "Prepare failed: " << sqlite3_errmsg(db_pointer) << endl;
        return nullopt;
    }

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        int id = sqlite3_column_int(stmt, 0);

        if (id!=id_to_confirm_with){
            return nullopt;
        }

        string description = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        float created_at = static_cast<float>(sqlite3_column_double(stmt, 2));

        cout << "Thread: " + to_string(id) + "\n"
        + "  Description: " + description + "\n"
        + "  Created At: " + to_string(created_at) + "\n";
        
        // ========================
        json result = ParseAndComputeData(description);
        // ========================

        sqlite3_finalize(stmt);

        const char* delete_sql = "DELETE FROM gpu_work_queue WHERE id = ?;";
        sqlite3_stmt* delete_stmt;
        if (sqlite3_prepare_v2(db_pointer, delete_sql, -1, &delete_stmt, nullptr) == SQLITE_OK) {
            sqlite3_bind_int(delete_stmt, 1, id);
            if (sqlite3_step(delete_stmt) != SQLITE_DONE) {
                cerr << "Delete failed: " << sqlite3_errmsg(db_pointer) << endl;
            } else {
                cout << "Deleted row ID: " << id << endl;
            }
            sqlite3_finalize(delete_stmt);
        } else {
            cerr << "Prepare delete failed: " << sqlite3_errmsg(db_pointer) << endl;
        }

        return result;
    } else {
        // cout << "No rows found.\n";
        // flush(cout);
        sqlite3_finalize(stmt);
        return nullopt;
    }
}

json keepchecking(int id, sqlite3* db_pointer) {
    int i = 0;
    while (true) {
        optional<json> output = read_from_sqlite3_db(db_pointer, id);
        if (output.has_value()) {
            return *output;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}