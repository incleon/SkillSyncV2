import re
import logging
from collections import defaultdict
from typing import Dict
import streamlit as st

logger = logging.getLogger(__name__)

# Try to load spaCy - make it optional for Streamlit deployment
try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    nlp = None
    st.warning("SpaCy model not found. Some NLP features will be limited.")


class AdvancedNLPProcessor:
    """Advanced NLP processing with NER, semantic analysis, and feature extraction"""

    def __init__(self):
        self.nlp = nlp

        # Enhanced skill categories with more comprehensive keywords
        self.skill_categories = {
            'programming_languages': {
                'keywords': ['python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust',
                             'kotlin', 'swift'],
                'weight': 1.5
            },
            'web_technologies': {
                'keywords': ['html', 'css', 'react', 'angular', 'vue', 'node', 'express', 'django', 'flask', 'spring'],
                'weight': 1.3
            },
            'databases': {
                'keywords': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle'],
                'weight': 1.4
            },
            'data_science': {
                'keywords': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy',
                             'scikit-learn', 'keras', 'opencv'],
                'weight': 1.6
            },
            'cloud_platforms': {
                'keywords': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'jenkins', 'ci/cd'],
                'weight': 1.4
            },
            'soft_skills': {
                'keywords': ['leadership', 'communication', 'teamwork', 'problem solving', 'analytical', 'creative'],
                'weight': 1.1
            },
            'methodologies': {
                'keywords': ['agile', 'scrum', 'kanban', 'devops', 'test driven development', 'microservices'],
                'weight': 1.2
            }
        }

        # Experience level patterns
        self.experience_patterns = {
            'years': [
                r'(\d+)[\+\s]*years?\s+(?:of\s+)?experience',
                r'(\d+)[\+\s]*yrs?\s+(?:of\s+)?experience',
                r'experience[:\s]*(\d+)[\+\s]*years?',
                r'(\d+)[\+\s]*years?\s+in',
                r'over\s+(\d+)\s+years?'
            ],
            'seniority': {
                'entry': ['entry level', 'junior', 'associate', 'trainee', 'intern', 'graduate'],
                'mid': ['mid level', 'intermediate', 'experienced', 'professional'],
                'senior': ['senior', 'lead', 'principal', 'architect', 'manager', 'director'],
                'executive': ['vp', 'vice president', 'cto', 'ceo', 'head of', 'chief']
            }
        }

        # Education patterns
        self.education_patterns = {
            'degree_types': ['bachelor', 'master', 'phd', 'doctorate', 'associates', 'diploma', 'certificate'],
            'fields': ['computer science', 'engineering', 'mathematics', 'statistics', 'physics', 'business']
        }

    def extract_named_entities(self, text: str) -> Dict:
        """Extract named entities using spaCy"""
        if not self.nlp:
            return {}

        try:
            doc = self.nlp(text)
            entities = {
                'organizations': [],
                'skills': [],
                'locations': [],
                'dates': [],
                'technologies': []
            }

            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities['organizations'].append(ent.text)
                elif ent.label_ == "GPE":
                    entities['locations'].append(ent.text)
                elif ent.label_ == "DATE":
                    entities['dates'].append(ent.text)

            return entities

        except Exception as e:
            logger.error(f"Error in NER extraction: {str(e)}")
            return {}

    def extract_skills_advanced(self, text: str) -> Dict:
        """Advanced skill extraction with categorization and confidence scoring"""
        try:
            text_lower = text.lower()
            skills_found = defaultdict(list)
            skill_confidence = {}

            for category, category_data in self.skill_categories.items():
                keywords = category_data['keywords']
                weight = category_data['weight']

                for keyword in keywords:
                    # Count occurrences and context
                    pattern = rf'\b{re.escape(keyword)}\b'
                    matches = re.findall(pattern, text_lower)

                    if matches:
                        count = len(matches)
                        confidence = min(count * 0.2, 1.0) * weight

                        skills_found[category].append({
                            'skill': keyword,
                            'count': count,
                            'confidence': confidence
                        })
                        skill_confidence[keyword] = confidence

            return {
                'categorized_skills': dict(skills_found),
                'skill_confidence': skill_confidence,
                'total_skills': len(skill_confidence)
            }

        except Exception as e:
            logger.error(f"Error in advanced skill extraction: {str(e)}")
            return {'categorized_skills': {}, 'skill_confidence': {}, 'total_skills': 0}

    def extract_experience_level(self, text: str) -> Dict:
        """Extract experience level information"""
        try:
            text_lower = text.lower()
            experience_info = {
                'years': 0,
                'level': 'entry',
                'confidence': 0.0
            }

            # Extract years of experience
            for pattern in self.experience_patterns['years']:
                matches = re.findall(pattern, text_lower)
                if matches:
                    years = max([int(match) for match in matches])
                    experience_info['years'] = years
                    break

            # Determine seniority level
            max_score = 0
            detected_level = 'entry'

            for level, keywords in self.experience_patterns['seniority'].items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > max_score:
                    max_score = score
                    detected_level = level

            experience_info['level'] = detected_level
            experience_info['confidence'] = min(max_score * 0.3, 1.0)

            # Adjust level based on years
            if experience_info['years'] >= 8:
                experience_info['level'] = 'senior'
            elif experience_info['years'] >= 4:
                experience_info['level'] = 'mid'
            elif experience_info['years'] >= 1:
                experience_info['level'] = 'entry'

            return experience_info

        except Exception as e:
            logger.error(f"Error extracting experience level: {str(e)}")
            return {'years': 0, 'level': 'entry', 'confidence': 0.0}

    def extract_education_info(self, text: str) -> Dict:
        """Extract education information"""
        try:
            text_lower = text.lower()
            education_info = {
                'highest_degree': 'none',
                'field_of_study': [],
                'institutions': [],
                'confidence': 0.0
            }

            # Extract degree types
            degree_scores = {}
            for degree in self.education_patterns['degree_types']:
                if degree in text_lower:
                    degree_scores[degree] = text_lower.count(degree)

            if degree_scores:
                highest_degree = max(degree_scores.keys(), key=lambda x: degree_scores[x])
                education_info['highest_degree'] = highest_degree
                education_info['confidence'] = min(degree_scores[highest_degree] * 0.4, 1.0)

            # Extract fields of study
            for field in self.education_patterns['fields']:
                if field in text_lower:
                    education_info['field_of_study'].append(field)

            # Extract institutions using NER if available
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG" and any(
                            word in ent.text.lower() for word in ['university', 'college', 'institute', 'school']):
                        education_info['institutions'].append(ent.text)

            return education_info

        except Exception as e:
            logger.error(f"Error extracting education info: {str(e)}")
            return {'highest_degree': 'none', 'field_of_study': [], 'institutions': [], 'confidence': 0.0}