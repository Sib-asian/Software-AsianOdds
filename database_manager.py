"""
Database Manager
================

Centralized database management for SQLite operations.
Provides connection pooling, transaction management, and safe query execution.
"""

import sqlite3
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database errors"""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails"""
    pass


class DatabaseQueryError(DatabaseError):
    """Raised when query execution fails"""
    pass


class DatabaseManager:
    """
    Centralized SQLite database manager with connection pooling and safe operations.

    Features:
    - Thread-safe connection management
    - Context manager support for transactions
    - Parameterized queries to prevent SQL injection
    - Automatic connection cleanup
    - Error handling and logging
    """

    def __init__(self, db_path: str, timeout: float = 5.0, check_same_thread: bool = False):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            timeout: Connection timeout in seconds
            check_same_thread: If False, allow connections across threads
        """
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        self._local = threading.local()

        # Create database directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use persistent disk on Render if available
        if Path('/data').exists() and Path('/data').is_dir():
            self.db_path = Path('/data') / self.db_path.name
            logger.info(f"📁 Using persistent disk: {self.db_path}")

        logger.info(f"🗄️  Database manager initialized: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            try:
                self._local.connection = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.timeout,
                    check_same_thread=self.check_same_thread,
                    isolation_level=None  # Autocommit mode
                )
                # Enable foreign keys
                self._local.connection.execute("PRAGMA foreign_keys = ON")
                # Use WAL mode for better concurrency
                self._local.connection.execute("PRAGMA journal_mode = WAL")
                # Return rows as dictionaries
                self._local.connection.row_factory = sqlite3.Row

            except sqlite3.Error as e:
                logger.error(f"❌ Failed to connect to database {self.db_path}: {e}")
                raise DatabaseConnectionError(f"Database connection failed: {e}") from e

        return self._local.connection

    def close_connection(self):
        """Close thread-local connection"""
        if hasattr(self._local, 'connection') and self._local.connection is not None:
            try:
                self._local.connection.close()
                self._local.connection = None
                logger.debug("Database connection closed")
            except sqlite3.Error as e:
                logger.error(f"Error closing connection: {e}")

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.

        Usage:
            with db_manager.transaction():
                db_manager.execute("INSERT INTO ...")
                db_manager.execute("UPDATE ...")
            # Commits automatically on success, rolls back on exception
        """
        conn = self._get_connection()
        conn.isolation_level = 'DEFERRED'  # Start transaction

        try:
            yield conn
            conn.commit()
            logger.debug("Transaction committed")
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
        finally:
            conn.isolation_level = None  # Back to autocommit

    def execute(
        self,
        query: str,
        params: Optional[Tuple[Any, ...]] = None,
        commit: bool = True
    ) -> sqlite3.Cursor:
        """
        Execute a SQL query with parameters.

        Args:
            query: SQL query with ? placeholders
            params: Query parameters tuple
            commit: Whether to commit after execution

        Returns:
            Cursor with query results

        Raises:
            DatabaseQueryError: If query execution fails
        """
        conn = self._get_connection()

        try:
            if params:
                cursor = conn.execute(query, params)
            else:
                cursor = conn.execute(query)

            if commit and conn.isolation_level is None:
                conn.commit()

            return cursor

        except sqlite3.Error as e:
            logger.error(f"❌ Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise DatabaseQueryError(f"Query execution failed: {e}") from e

    def execute_many(
        self,
        query: str,
        params_list: List[Tuple[Any, ...]],
        commit: bool = True
    ) -> sqlite3.Cursor:
        """
        Execute a SQL query multiple times with different parameters.

        Args:
            query: SQL query with ? placeholders
            params_list: List of parameter tuples
            commit: Whether to commit after execution

        Returns:
            Cursor object
        """
        conn = self._get_connection()

        try:
            cursor = conn.executemany(query, params_list)

            if commit and conn.isolation_level is None:
                conn.commit()

            return cursor

        except sqlite3.Error as e:
            logger.error(f"❌ Batch execution failed: {e}")
            raise DatabaseQueryError(f"Batch execution failed: {e}") from e

    def fetch_one(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> Optional[Dict[str, Any]]:
        """
        Execute query and fetch one result as dictionary.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Dictionary with column names as keys, or None if no results
        """
        cursor = self.execute(query, params, commit=False)
        row = cursor.fetchone()
        return dict(row) if row else None

    def fetch_all(self, query: str, params: Optional[Tuple[Any, ...]] = None) -> List[Dict[str, Any]]:
        """
        Execute query and fetch all results as list of dictionaries.

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            List of dictionaries with column names as keys
        """
        cursor = self.execute(query, params, commit=False)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def create_table(self, table_name: str, schema: str):
        """
        Create a table if it doesn't exist.

        Args:
            table_name: Name of the table
            schema: SQL schema definition (column definitions)
        """
        query = f"CREATE TABLE IF NOT EXISTS {table_name} ({schema})"
        self.execute(query)
        logger.info(f"✅ Table '{table_name}' ensured to exist")

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists, False otherwise
        """
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
        result = self.fetch_one(query, (table_name,))
        return result is not None

    def count_rows(self, table_name: str, where: Optional[str] = None, params: Optional[Tuple[Any, ...]] = None) -> int:
        """
        Count rows in a table.

        Args:
            table_name: Name of the table
            where: Optional WHERE clause (without WHERE keyword)
            params: Parameters for WHERE clause

        Returns:
            Number of rows
        """
        query = f"SELECT COUNT(*) as count FROM {table_name}"
        if where:
            query += f" WHERE {where}"

        result = self.fetch_one(query, params)
        return result['count'] if result else 0

    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get information about table columns.

        Args:
            table_name: Name of the table

        Returns:
            List of dictionaries with column information
        """
        query = f"PRAGMA table_info({table_name})"
        return self.fetch_all(query)

    def vacuum(self):
        """Optimize database by running VACUUM"""
        try:
            conn = self._get_connection()
            conn.execute("VACUUM")
            logger.info("✅ Database vacuumed successfully")
        except sqlite3.Error as e:
            logger.error(f"❌ Failed to vacuum database: {e}")

    def get_database_size(self) -> int:
        """
        Get database file size in bytes.

        Returns:
            Size in bytes, or 0 if file doesn't exist
        """
        if self.db_path.exists():
            return self.db_path.stat().st_size
        return 0

    def get_database_size_mb(self) -> float:
        """
        Get database file size in megabytes.

        Returns:
            Size in MB
        """
        return self.get_database_size() / (1024 * 1024)

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection"""
        self.close_connection()

    def __del__(self):
        """Cleanup on deletion"""
        self.close_connection()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def get_db_manager(db_name: str = "api_cache.db") -> DatabaseManager:
    """
    Get a database manager instance.

    Args:
        db_name: Database filename

    Returns:
        DatabaseManager instance
    """
    return DatabaseManager(db_name)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Example usage
    db = DatabaseManager("test.db")

    # Create a table
    db.create_table(
        "users",
        """
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """
    )

    # Insert data (parameterized query - safe from SQL injection)
    db.execute(
        "INSERT INTO users (name, email) VALUES (?, ?)",
        ("John Doe", "john@example.com")
    )

    # Fetch data
    users = db.fetch_all("SELECT * FROM users")
    print(f"Users: {users}")

    # Use transaction
    with db.transaction():
        db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Jane Doe", "jane@example.com"))
        db.execute("INSERT INTO users (name, email) VALUES (?, ?)", ("Bob Smith", "bob@example.com"))

    # Count rows
    count = db.count_rows("users")
    print(f"Total users: {count}")

    # Database info
    print(f"Database size: {db.get_database_size_mb():.2f} MB")

    # Cleanup
    db.close_connection()
