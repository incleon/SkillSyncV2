import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List
import pickle
import os

logger = logging.getLogger(__name__)


class MLEnhancedMatcher:
    """Machine Learning enhanced matching with clustering and classification"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False

        # Try to load pre-trained models
        self._load_models()

    def _load_models(self):
        """Load pre-trained models if available"""
        try:
            if os.path.exists('models/classifier.pkl'):
                with open('models/classifier.pkl', 'rb') as f:
                    self.classifier = pickle.load(f)
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Pre-trained models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {str(e)}")

    def _save_models(self):
        """Save trained models"""
        try:
            os.makedirs('models', exist_ok=True)
            with open('models/classifier.pkl', 'wb') as f:
                pickle.dump(self.classifier, f)
            with open('models/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")

    def extract_ml_features(self, job_text: str, resume_text: str, skills_data: Dict,
                            experience_data: Dict, education_data: Dict) -> np.ndarray:
        """Extract comprehensive features for ML models"""
        try:
            features = []

            # Basic text features
            features.extend([
                len(job_text.split()),
                len(resume_text.split()),
                len(set(job_text.lower().split()) & set(resume_text.lower().split())),
            ])

            # Skills features
            total_skills = skills_data.get('total_skills', 0)
            avg_confidence = np.mean(list(skills_data.get('skill_confidence', {0: 0}).values())) if skills_data.get(
                'skill_confidence') else 0
            features.extend([total_skills, avg_confidence])

            # Experience features
            features.extend([
                experience_data.get('years', 0),
                1 if experience_data.get('level') == 'senior' else 0,
                1 if experience_data.get('level') == 'mid' else 0,
                experience_data.get('confidence', 0)
            ])

            # Education features
            education_score = {
                'phd': 5, 'doctorate': 5,
                'master': 4,
                'bachelor': 3,
                'associates': 2,
                'diploma': 1,
                'certificate': 1,
                'none': 0
            }.get(education_data.get('highest_degree', 'none'), 0)

            features.extend([
                education_score,
                len(education_data.get('field_of_study', [])),
                education_data.get('confidence', 0)
            ])

            # Text similarity features using TF-IDF
            try:
                vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2), stop_words='english')
                tfidf_matrix = vectorizer.fit_transform([job_text, resume_text])
                tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                tfidf_sim = 0.0

            features.append(tfidf_sim)

            return np.array(features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Error extracting ML features: {str(e)}")
            return np.zeros((1, 14))

    def predict_match_probability(self, features: np.ndarray) -> float:
        """Predict match probability using trained classifier or simple heuristic"""
        try:
            if self.is_trained:
                scaled_features = self.scaler.transform(features)
                probabilities = self.classifier.predict_proba(scaled_features)
                return float(probabilities[0][1]) if probabilities.shape[1] > 1 else float(probabilities[0][0])
            else:
                # Simple heuristic based on features
                feature_vector = features[0]
                # Use TF-IDF similarity and normalize other features
                tfidf_score = feature_vector[-1]  # Last feature is TF-IDF
                skills_score = min(feature_vector[4] / 10, 1.0) if len(feature_vector) > 4 else 0  # Skills confidence
                experience_score = min(feature_vector[5] / 10, 1.0) if len(
                    feature_vector) > 5 else 0  # Years experience

                return (tfidf_score * 0.5 + skills_score * 0.3 + experience_score * 0.2)

        except Exception as e:
            logger.error(f"Error predicting match probability: {str(e)}")
            return 0.0

    def detect_anomalies(self, features_list: List[np.ndarray]) -> List[bool]:
        """Detect anomalous resumes"""
        try:
            if len(features_list) < 2:
                return [False] * len(features_list)

            combined_features = np.vstack(features_list)
            anomaly_scores = self.anomaly_detector.fit_predict(combined_features)

            return [score == -1 for score in anomaly_scores]

        except Exception as e:
            logger.error(f"Error detecting anomalies: {str(e)}")
            return [False] * len(features_list)