import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict

logger = logging.getLogger(__name__)

# Try to import sentence transformers - make it optional
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SemanticMatcher:
    """Advanced semantic matching using transformer models"""

    def __init__(self):
        self.sentence_model = None
        self.model_loaded = False

        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Load pre-trained sentence transformer
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model_loaded = True
                logger.info("Sentence transformer model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {str(e)}")
                self.model_loaded = False

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using sentence transformers or TF-IDF fallback"""
        try:
            if self.model_loaded and text1 and text2:
                # Use sentence transformers if available
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            else:
                # Fallback to TF-IDF
                return self._get_tfidf_similarity(text1, text2)

        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return self._get_tfidf_similarity(text1, text2)

    def _get_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Fallback TF-IDF similarity calculation"""
        try:
            if not text1 or not text2:
                return 0.0

            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {str(e)}")
            return 0.0

    def get_section_similarities(self, job_desc: str, resume_text: str) -> Dict:
        """Calculate similarities for different sections"""
        try:
            # Split into sections (basic approach)
            job_sections = self._split_into_sections(job_desc)
            resume_sections = self._split_into_sections(resume_text)

            section_similarities = {}

            for job_section, job_content in job_sections.items():
                if job_content and job_section in resume_sections and resume_sections[job_section]:
                    similarity = self.get_semantic_similarity(job_content, resume_sections[job_section])
                    section_similarities[job_section] = similarity

            return section_similarities

        except Exception as e:
            logger.error(f"Error calculating section similarities: {str(e)}")
            return {}

    def _split_into_sections(self, text: str) -> Dict:
        """Basic section splitting"""
        sections = {
            'skills': '',
            'experience': '',
            'education': '',
            'summary': ''
        }

        text_lower = text.lower()

        # Simple keyword-based section detection
        if any(word in text_lower for word in ['skills', 'technical', 'technologies']):
            sections['skills'] = text[:200]

        if any(word in text_lower for word in ['experience', 'work', 'employment']):
            sections['experience'] = text[:300]

        if any(word in text_lower for word in ['education', 'degree', 'university']):
            sections['education'] = text[:200]

        sections['summary'] = text[:150]

        return sections