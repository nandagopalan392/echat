import sqlite3
import json
import logging
import hashlib  # Add this import at the top
from datetime import datetime
import os
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class ChatDB:
    def __init__(self, db_path=None):
        if db_path is None:
            # Use environment variable with default path
            db_dir = os.getenv('SQLITE_DB_PATH', '/app/data/db')
            db_dir = Path(db_dir).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            self.db_path = os.path.join(db_dir, 'chat.db')
        else:
            self.db_path = db_path
        
        # Ensure directory exists and has proper permissions
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        db_dir.chmod(0o777)
        
        logger.info(f"Using database path: {self.db_path}")
        self.init_db()

    def init_db(self):
        try:
            # Only create tables if they don't exist
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if users table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='users'
                """)
                if not cursor.fetchone():
                    # Create users table and admin user only if table doesn't exist
                    cursor.execute('''
                        CREATE TABLE users (
                            username TEXT PRIMARY KEY,
                            password_hash TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            is_admin BOOLEAN DEFAULT FALSE,
                            role TEXT DEFAULT 'Engineer'
                        )
                    ''')
                    
                    # Create admin user
                    admin_password = 'admin123'
                    hashed_password = hashlib.sha256(admin_password.encode()).hexdigest()
                    cursor.execute(
                        'INSERT INTO users (username, password_hash, is_admin, role) VALUES (?, ?, ?, ?)',
                        ('admin', hashed_password, True, 'Admin')
                    )
                    logger.info("Created admin user")

                # Create other tables if they don't exist
                cursor.execute('''CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    format TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    uploaded_by TEXT NOT NULL,
                    is_folder BOOLEAN DEFAULT FALSE,
                    folder_path TEXT,
                    FOREIGN KEY (uploaded_by) REFERENCES users (username)
                )''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        topic TEXT,
                        created_at TIMESTAMP,
                        last_updated TIMESTAMP
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        content TEXT,
                        is_user BOOLEAN,
                        timestamp TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
                    )
                ''')

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS response_cache (
                        id INTEGER PRIMARY KEY,
                        query_hash TEXT UNIQUE,
                        query TEXT,
                        response TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()
                logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise

    def create_session(self, username, first_message=""):
        start_time = time.time()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                # Create a meaningful topic from the first message
                topic = (first_message[:50] + '...') if len(first_message) > 50 else first_message
                if not topic.strip():
                    topic = "New Chat"
                    
                cursor.execute(
                    """INSERT INTO chat_sessions 
                       (username, topic, created_at, last_updated) 
                       VALUES (?, ?, ?, ?)""",
                    (username, topic, now, now)
                )
                conn.commit()
                session_id = cursor.lastrowid
                logger.debug(f"Created new session with ID {session_id} and topic: {topic}")
                return session_id
        except Exception as e:
            logger.error(f"Create session error: {str(e)}")
            raise
        finally:
            duration = time.time() - start_time
            # Log database operation metrics instead of using Prometheus
            logger.debug(f"DB operation - create chat_sessions took {duration:.3f}s")

    def update_session_topic(self, session_id, new_topic):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE chat_sessions SET topic = ? WHERE id = ?",
                    (new_topic, session_id)
                )
        except Exception as e:
            logger.error(f"Update session topic error: {str(e)}")

    def save_message(self, session_id, content, is_user):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                cursor.execute(
                    "INSERT INTO messages (session_id, content, is_user, timestamp) VALUES (?, ?, ?, ?)",
                    (session_id, content, is_user, now)
                )
                cursor.execute(
                    "UPDATE chat_sessions SET last_updated = ? WHERE id = ?",
                    (now, session_id)
                )
        except Exception as e:
            logger.error(f"Save message error: {str(e)}")

    def update_message(self, message_id, content):
        """Update an existing message content"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                now = datetime.now()
                
                # Update the message content and timestamp
                cursor.execute(
                    "UPDATE messages SET content = ?, timestamp = ? WHERE id = ?",
                    (content, now, message_id)
                )
                
                # Check if any rows were affected
                if cursor.rowcount == 0:
                    logger.warning(f"No message found with id {message_id}")
                    return False
                
                # Update the session's last_updated timestamp
                cursor.execute(
                    """UPDATE chat_sessions SET last_updated = ? 
                       WHERE id = (SELECT session_id FROM messages WHERE id = ?)""",
                    (now, message_id)
                )
                
                logger.info(f"Message {message_id} updated successfully")
                return True
        except Exception as e:
            logger.error(f"Update message error: {str(e)}")
            return False

    def get_latest_ai_message_id(self, session_id):
        """Get the ID of the latest AI message in a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT id FROM messages 
                       WHERE session_id = ? AND is_user = 0 
                       ORDER BY timestamp DESC LIMIT 1""",
                    (session_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Get latest AI message ID error: {str(e)}")
            return None

    def get_user_sessions(self, username):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT cs.id, cs.created_at, 
                              COALESCE(cs.topic, 
                                      (SELECT content FROM messages 
                                       WHERE session_id = cs.id AND is_user = 1 
                                       ORDER BY timestamp ASC LIMIT 1)
                              ) as topic
                       FROM chat_sessions cs
                       WHERE username = ? 
                       ORDER BY last_updated DESC""",
                    (username,)
                )
                sessions = []
                for row in cursor.fetchall():
                    topic = row['topic'] if row['topic'] else "New Chat"
                    sessions.append({
                        "id": row['id'],
                        "date": row['created_at'],
                        "topic": topic
                    })
                return sessions
        except Exception as e:
            logger.error(f"Get user sessions error: {str(e)}")
            return []

    def get_session_messages(self, session_id):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """SELECT content, is_user, timestamp 
                       FROM messages 
                       WHERE session_id = ? 
                       ORDER BY timestamp ASC, id ASC""",
                    (session_id,)
                )
                messages = cursor.fetchall()
                logger.debug(f"Retrieved {len(messages)} messages for session {session_id}")
                return messages
        except Exception as e:
            logger.error(f"Get session messages error: {str(e)}")
            return []

    def add_user(self, username: str, password: str, role: str = 'Engineer') -> bool:
        try:
            if role not in ['Engineer', 'Manager', 'Business Development', 'Associate']:
                raise ValueError(f"Invalid role: {role}. Must be Engineer, Manager, Business Development, or Associate")
                
            # Use simple SHA-256 hash for consistency
            hashed_password = hashlib.sha256(password.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO users (username, password_hash, role, is_admin) VALUES (?, ?, ?, ?)',
                    (username, hashed_password, role, False)
                )
                conn.commit()
                logger.info(f"Successfully added user: {username} with role: {role}")
                return True
                
        except sqlite3.IntegrityError as e:
            logger.error(f"Database integrity error adding user: {str(e)}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error adding user: {str(e)}", exc_info=True)
            raise

    def authenticate_user(self, username: str, password: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                logger.debug(f"Authenticating user: {username}")
                
                cursor.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
                result = cursor.fetchone()
                
                if not result:
                    logger.warning(f"User not found: {username}")
                    return False
                
                stored_password_hash = result[0]
                # Use simple SHA-256 hash comparison
                provided_hash = hashlib.sha256(password.encode()).hexdigest()
                is_valid = stored_password_hash == provided_hash
                
                logger.debug(f"Authentication result for {username}: {is_valid}")
                return is_valid
                
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}", exc_info=True)
            return False

    def get_user(self, username):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT username, password_hash FROM users WHERE username = ?',
                               (username,))
                return cursor.fetchone()
        except Exception as e:
            logger.error(f"Get user error: {str(e)}")
            return None

    def user_exists(self, username):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1 FROM users WHERE username = ?', (username,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"User exists check error: {str(e)}")
            return False

    def is_admin(self, username: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT is_admin FROM users WHERE username = ?', (username,))
                result = cursor.fetchone()
                return bool(result and result[0])
        except Exception as e:
            logger.error(f"Admin check error: {str(e)}")
            return False

    def get_all_users(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT username, is_admin, role FROM users')
                return [{"username": row[0], "is_admin": row[1], "role": row[2]} for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Get users error: {str(e)}")
            return []

    def get_user_stats(self, username: str):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total messages and last message
                cursor.execute("""
                    SELECT COUNT(*), MAX(m.timestamp), m.content
                    FROM messages m
                    JOIN chat_sessions cs ON m.session_id = cs.id
                    WHERE cs.username = ?
                """, (username,))
                messages_data = cursor.fetchone()
                total_messages = messages_data[0]
                last_active = messages_data[1]
                last_message = messages_data[2]

                # Get total sessions
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM chat_sessions
                    WHERE username = ?
                """, (username,))
                total_sessions = cursor.fetchone()[0]

                # Get recent chats with their latest messages
                cursor.execute("""
                    SELECT cs.topic, cs.last_updated,
                           (SELECT content FROM messages 
                            WHERE session_id = cs.id 
                            ORDER BY timestamp DESC LIMIT 1) as last_message
                    FROM chat_sessions cs
                    WHERE cs.username = ?
                    ORDER BY cs.last_updated DESC
                    LIMIT 5
                """, (username,))
                
                recent_chats = [{
                    "topic": row[0] or "New Chat",
                    "date": row[1],
                    "lastMessage": row[2] or "No messages"
                } for row in cursor.fetchall()]

                return {
                    "totalMessages": total_messages,
                    "totalSessions": total_sessions,
                    "lastActive": last_active,
                    "lastMessage": last_message,
                    "recentChats": recent_chats
                }
        except Exception as e:
            logger.error(f"Get user stats error: {str(e)}")
            raise

    def get_activity_stats(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total stats
                cursor.execute("SELECT COUNT(*) FROM users WHERE is_admin = 0")
                total_users = cursor.fetchone()[0]

                cursor.execute("""
                    SELECT COUNT(DISTINCT username) 
                    FROM chat_sessions 
                    WHERE DATE(last_updated) = DATE('now')
                """)
                active_users = cursor.fetchone()[0]

                cursor.execute("SELECT COUNT(*) FROM messages")
                total_messages = cursor.fetchone()[0]

                # Get recent activities
                cursor.execute("""
                    SELECT cs.username, m.content, m.timestamp
                    FROM messages m
                    JOIN chat_sessions cs ON m.session_id = cs.id
                    ORDER BY m.timestamp DESC
                    LIMIT 10
                """)
                activities = [{
                    "username": row[0],
                    "action": "sent a message",
                    "timestamp": row[2]
                } for row in cursor.fetchall()]

                return {
                    "totalUsers": total_users,
                    "activeUsers": active_users,
                    "totalMessages": total_messages,
                    "recentActivities": activities
                }
        except Exception as e:
            logger.error(f"Get activity stats error: {str(e)}")
            raise

    def save_file_info(self, filename, format, size, uploaded_by, is_folder=False, folder_path=None):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'INSERT INTO files (filename, format, size, uploaded_by, is_folder, folder_path) VALUES (?, ?, ?, ?, ?, ?)',
                    (filename, format, size, uploaded_by, is_folder, folder_path)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Save file info error: {str(e)}")
            raise

    def get_all_files(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT filename, format, size, upload_date, uploaded_by
                    FROM files
                    ORDER BY upload_date DESC
                ''')
                files = cursor.fetchall()
                return [{
                    'filename': f[0],
                    'format': f[1],
                    'size': f[2],
                    'upload_date': f[3],
                    'uploaded_by': f[4]
                } for f in files]
        except Exception as e:
            logger.error(f"Get files error: {str(e)}")
            return []

    def get_file_stats(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total counts and sizes
                cursor.execute("""
                    SELECT COUNT(*) as total_files,
                           SUM(size) as total_size,
                           format,
                           COUNT(*) as format_count
                    FROM files
                    GROUP BY format
                """)
                
                format_stats = cursor.fetchall()
                total_files = sum(row[3] for row in format_stats)
                total_size = sum(row[1] for row in format_stats)
                
                return {
                    "totalFiles": total_files,
                    "totalSize": total_size,
                    "formatStats": [{
                        "format": row[2],
                        "count": row[3],
                        "totalSize": row[1]
                    } for row in format_stats]
                }
                
        except Exception as e:
            logger.error(f"Get file stats error: {str(e)}")
            return {
                "totalFiles": 0,
                "totalSize": 0,
                "formatStats": []
            }

    def get_cached_response(self, query):
        """Get a cached response for a query if it exists and is fresh"""
        try:
            # Create a hash key for the query
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            # Get the cached response if it exists
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT response, timestamp FROM response_cache
                    WHERE query_hash = ? AND 
                          datetime('now', '-1 hour') < datetime(timestamp)
                """, (query_hash,))
                result = cursor.fetchone()
                
                if result:
                    return result[0]
                return None
        except Exception as e:
            logger.error(f"Error getting cached response: {str(e)}")
            return None

    def cache_response(self, query, response):
        """Cache a response for a query"""
        try:
            # Create a hash key for the query
            import hashlib
            query_hash = hashlib.md5(query.encode()).hexdigest()
            
            # Store the response in the cache
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO response_cache
                    (query_hash, query, response, timestamp)
                    VALUES (?, ?, ?, datetime('now'))
                """, (query_hash, query, response))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error caching response: {str(e)}")
            return False

    def _get_connection(self):
        """Get a connection to the database with error handling"""
        try:
            # Make sure the database directory exists
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            # Return a more informative error
            raise RuntimeError(f"Could not connect to database at {self.db_path}: {str(e)}")

    # Add these enhanced DB monitoring methods

    def get_active_sessions(self):
        """Get active sessions for monitoring purposes"""
        start_time = time.time()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Get sessions active in the last hour
                cursor.execute("""
                    SELECT username, created_at, last_updated
                    FROM chat_sessions
                    WHERE datetime(last_updated) > datetime('now', '-1 hour')
                    ORDER BY last_updated DESC
                """)
                sessions = cursor.fetchall()
                
                # Log session count instead of setting Prometheus metric
                users = {}
                for session in sessions:
                    username = session[0]
                    if username in users:
                        users[username] += 1
                    else:
                        users[username] = 1
                
                # Log session count per user
                for username, count in users.items():
                    logger.debug(f"Active sessions for {username}: {count}")
                
                return sessions
        except Exception as e:
            logger.error(f"Error getting active sessions: {str(e)}")
            return []
        finally:
            duration = time.time() - start_time
            logger.debug(f"DB operation - select chat_sessions took {duration:.3f}s")

    def get_cache_stats(self):
        """Get cache statistics for monitoring"""
        start_time = time.time()
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Get total cache entries
                cursor.execute("SELECT COUNT(*) FROM response_cache")
                total = cursor.fetchone()[0]
                
                # Get fresh cache entries (less than 1 hour old)
                cursor.execute("SELECT COUNT(*) FROM response_cache WHERE datetime(timestamp) > datetime('now', '-1 hour')")
                fresh = cursor.fetchone()[0]
                
                # Get cache size in bytes
                cursor.execute("SELECT SUM(length(response)) FROM response_cache")
                size = cursor.fetchone()[0] or 0
                
                # Update cache size metric if it exists
                if CACHE_SIZE:
                    CACHE_SIZE.set(total)
                
                return {
                    "total_entries": total,
                    "fresh_entries": fresh,
                    "size_bytes": size
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"total_entries": 0, "fresh_entries": 0, "size_bytes": 0}
        finally:
            duration = time.time() - start_time
            logger.debug(f"DB operation - cache_stats took {duration:.3f}s")
