import sqlite3
import logging
import json
import os

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages SQLite database for storing historical data and feedback"""

    def __init__(self, db_path):
        self.db_path = db_path
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Job descriptions table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS job_descriptions
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               content
                               TEXT
                               NOT
                               NULL,
                               created_at
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               industry
                               TEXT,
                               experience_level
                               TEXT,
                               skills_required
                               TEXT
                           )
                           ''')

            # Resumes table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS resumes
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               filename
                               TEXT
                               NOT
                               NULL,
                               content
                               TEXT
                               NOT
                               NULL,
                               created_at
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               skills_extracted
                               TEXT,
                               experience_years
                               INTEGER,
                               education_level
                               TEXT
                           )
                           ''')

            # Matching results table
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS matching_results
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               job_id
                               INTEGER,
                               resume_id
                               INTEGER,
                               similarity_score
                               REAL,
                               semantic_score
                               REAL,
                               skills_score
                               REAL,
                               experience_score
                               REAL,
                               created_at
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               FOREIGN
                               KEY
                           (
                               job_id
                           ) REFERENCES job_descriptions
                           (
                               id
                           ),
                               FOREIGN KEY
                           (
                               resume_id
                           ) REFERENCES resumes
                           (
                               id
                           )
                               )
                           ''')

            # Feedback table for learning
            cursor.execute('''
                           CREATE TABLE IF NOT EXISTS hiring_feedback
                           (
                               id
                               INTEGER
                               PRIMARY
                               KEY
                               AUTOINCREMENT,
                               job_id
                               INTEGER,
                               resume_id
                               INTEGER,
                               hired
                               BOOLEAN,
                               interview_called
                               BOOLEAN,
                               feedback_date
                               TIMESTAMP
                               DEFAULT
                               CURRENT_TIMESTAMP,
                               FOREIGN
                               KEY
                           (
                               job_id
                           ) REFERENCES job_descriptions
                           (
                               id
                           ),
                               FOREIGN KEY
                           (
                               resume_id
                           ) REFERENCES resumes
                           (
                               id
                           )
                               )
                           ''')

            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")

    def store_job_description(self, content, industry=None, experience_level=None, skills_required=None):
        """Store job description and return ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO job_descriptions (content, industry, experience_level, skills_required)
                           VALUES (?, ?, ?, ?)
                           ''', (content, industry, experience_level,
                                 json.dumps(skills_required) if skills_required else None))

            job_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return job_id

        except Exception as e:
            logger.error(f"Error storing job description: {str(e)}")
            return None

    def store_resume(self, filename, content, skills_extracted=None, experience_years=None, education_level=None):
        """Store resume and return ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO resumes (filename, content, skills_extracted, experience_years, education_level)
                           VALUES (?, ?, ?, ?, ?)
                           ''', (filename, content, json.dumps(skills_extracted) if skills_extracted else None,
                                 experience_years, education_level))

            resume_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return resume_id

        except Exception as e:
            logger.error(f"Error storing resume: {str(e)}")
            return None

    def store_matching_result(self, job_id, resume_id, similarity_score, semantic_score, skills_score,
                              experience_score):
        """Store matching result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           INSERT INTO matching_results (job_id, resume_id, similarity_score, semantic_score,
                                                         skills_score, experience_score)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ''', (job_id, resume_id, similarity_score, semantic_score, skills_score, experience_score))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing matching result: {str(e)}")

    def get_historical_data(self, limit=100):
        """Get historical matching data for analytics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                           SELECT mr.*, jd.industry, jd.experience_level, r.filename
                           FROM matching_results mr
                                    JOIN job_descriptions jd ON mr.job_id = jd.id
                                    JOIN resumes r ON mr.resume_id = r.id
                           ORDER BY mr.created_at DESC LIMIT ?
                           ''', (limit,))

            results = cursor.fetchall()
            conn.close()

            return results

        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return []