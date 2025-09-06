import re
import logging
from collections import defaultdict
from typing import Dict, List
import streamlit as st

logger = logging.getLogger(__name__)

# Try to load spaCy - make it optional for Streamlit deployment
try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        logger.info("SpaCy model loaded successfully")
    except OSError:
        nlp = None
        SPACY_AVAILABLE = False
        st.warning(
            "⚠️ SpaCy model 'en_core_web_sm' not found. Using basic NLP processing. For better results, run: `python -m spacy download en_core_web_sm`")
except ImportError:
    nlp = None
    SPACY_AVAILABLE = False
    st.warning("⚠️ SpaCy not installed. Some NLP features will be limited.")


class AdvancedNLPProcessor:
    """Advanced NLP processing with fallback options when models are unavailable"""

    def __init__(self):
        self.nlp = nlp if SPACY_AVAILABLE else None

        # Enhanced skill categories with more comprehensive keywords
        self.skill_categories = {
            'programming_languages': {
                'keywords': [
                    'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'csharp',
                    'ruby', 'php', 'go', 'rust', 'kotlin', 'swift', 'scala', 'perl',
                    'r', 'matlab', 'sql', 'nosql', 'html', 'css'
                ],
                'weight': 1.5
            },
            'web_technologies': {
                'keywords': [
                    'react', 'angular', 'vue', 'node', 'nodejs', 'express', 'django',
                    'flask', 'spring', 'laravel', 'bootstrap', 'jquery', 'webpack',
                    'babel', 'sass', 'less', 'graphql', 'rest api', 'api'
                ],
                'weight': 1.3
            },
            'databases': {
                'keywords': [
                    'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                    'cassandra', 'oracle', 'sqlite', 'dynamodb', 'neo4j',
                    'database design', 'data modeling'
                ],
                'weight': 1.4
            },
            'data_science': {
                'keywords': [
                    'machine learning', 'deep learning', 'tensorflow', 'pytorch',
                    'pandas', 'numpy', 'scikit-learn', 'keras', 'opencv',
                    'data analysis', 'statistics', 'visualization', 'tableau',
                    'power bi', 'matplotlib', 'seaborn', 'plotly'
                ],
                'weight': 1.6
            },
            'cloud_platforms': {
                'keywords': [
                    'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes',
                    'terraform', 'jenkins', 'ci/cd', 'devops', 'serverless',
                    'lambda', 'ec2', 's3', 'cloudformation'
                ],
                'weight': 1.4
            },
            'soft_skills': {
                'keywords': [
                    'leadership', 'communication', 'teamwork', 'collaboration',
                    'problem solving', 'analytical', 'creative', 'adaptable',
                    'time management', 'project management', 'mentoring'
                ],
                'weight': 1.1
            },
            'methodologies': {
                'keywords': [
                    'agile', 'scrum', 'kanban', 'waterfall', 'lean',
                    'test driven development', 'tdd', 'bdd', 'microservices',
                    'mvp', 'design patterns', 'solid principles'
                ],
                'weight': 1.2
            },
            'tools_and_frameworks': {
                'keywords': [
                    'git', 'github', 'gitlab', 'jira', 'confluence', 'slack',
                    'visual studio', 'intellij', 'eclipse', 'postman',
                    'swagger', 'figma', 'sketch', 'adobe creative suite'
                ],
                'weight': 1.0
            }
        }

        # Enhanced experience patterns
        self.experience_patterns = {
            'years': [
                r'(\d+)[\+\s]*years?\s+(?:of\s+)?experience',
                r'(\d+)[\+\s]*yrs?\s+(?:of\s+)?experience',
                r'experience[:\s]*(\d+)[\+\s]*years?',
                r'(\d+)[\+\s]*years?\s+in',
                r'over\s+(\d+)\s+years?',
                r'(\d+)[\+\s]*year\s+experience',
                r'(\d{1,2})\+?\s*years',
                r'(\d+)\s*to\s*(\d+)\s*years'
            ],
            'seniority': {
                'entry': [
                    'entry level', 'junior', 'associate', 'trainee', 'intern',
                    'graduate', 'fresher', 'beginner', 'new grad'
                ],
                'mid': [
                    'mid level', 'intermediate', 'experienced', 'professional',
                    'specialist', 'analyst', 'consultant'
                ],
                'senior': [
                    'senior', 'lead', 'principal', 'architect', 'manager',
                    'team lead', 'tech lead', 'staff', 'expert'
                ],
                'executive': [
                    'director', 'vp', 'vice president', 'cto', 'ceo', 'coo',
                    'head of', 'chief', 'executive', 'president'
                ]
            }
        }

        # Enhanced education patterns
        self.education_patterns = {
            'degree_types': [
                'phd', 'doctorate', 'doctoral', 'master', 'masters', 'mba',
                'bachelor', 'bachelors', 'associates', 'diploma', 'certificate',
                'certification', 'degree'
            ],
            'fields': [
                'computer science', 'software engineering', 'information technology',
                'electrical engineering', 'mechanical engineering', 'mathematics',
                'statistics', 'physics', 'business', 'management', 'economics',
                'data science', 'artificial intelligence', 'cybersecurity'
            ]
        }

    def extract_named_entities(self, text: str) -> Dict:
        """Extract named entities using spaCy or fallback regex"""
        if self.nlp:
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
                logger.error(f"Error in spaCy NER extraction: {str(e)}")

        # Fallback to regex-based extraction
        return self._extract_entities_regex(text)

    def _extract_entities_regex(self, text: str) -> Dict:
        """Fallback regex-based entity extraction"""
        entities = {
            'organizations': [],
            'skills': [],
            'locations': [],
            'dates': [],
            'technologies': []
        }

        try:
            # Extract years (basic date extraction)
            year_pattern = r'\b(19|20)\d{2}\b'
            entities['dates'] = re.findall(year_pattern, text)

            # Extract potential company names (words followed by Inc, LLC, Corp, etc.)
            company_pattern = r'\b[A-Z][a-zA-Z\s&]+(?:Inc|LLC|Corp|Corporation|Company|Ltd|Limited)\b'
            entities['organizations'] = re.findall(company_pattern, text)

            # Extract cities (capitalized words that might be locations)
            location_pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*,\s*[A-Z]{2}\b'
            entities['locations'] = re.findall(location_pattern, text)

        except Exception as e:
            logger.error(f"Error in regex entity extraction: {str(e)}")

        return entities

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
                    # More flexible pattern matching
                    patterns = [
                        rf'\b{re.escape(keyword)}\b',  # Exact match
                        rf'{re.escape(keyword)}(?:\s*\d+(?:\.\d+)?)?',  # With version numbers
                        rf'{re.escape(keyword)}(?:\s*(?:js|py|dev|development))?'  # Common suffixes
                    ]

                    total_matches = 0
                    for pattern in patterns:
                        matches = re.findall(pattern, text_lower, re.IGNORECASE)
                        total_matches += len(matches)

                    if total_matches > 0:
                        # Enhanced confidence calculation
                        base_confidence = min(total_matches * 0.3, 1.0)
                        context_bonus = self._get_context_bonus(text_lower, keyword)
                        confidence = min((base_confidence + context_bonus) * weight, 1.0)

                        skills_found[category].append({
                            'skill': keyword,
                            'count': total_matches,
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

    def _get_context_bonus(self, text: str, skill: str) -> float:
        """Calculate bonus based on context around skill mentions"""
        try:
            skill_contexts = [
                'proficient', 'expert', 'experienced', 'skilled',
                'advanced', 'years', 'strong', 'excellent'
            ]

            bonus = 0.0
            skill_positions = [m.start() for m in re.finditer(rf'\b{re.escape(skill)}\b', text)]

            for pos in skill_positions:
                # Check context window around skill mention
                context_start = max(0, pos - 100)
                context_end = min(len(text), pos + 100)
                context = text[context_start:context_end]

                context_score = sum(1 for ctx in skill_contexts if ctx in context)
                bonus += context_score * 0.1

            return min(bonus, 0.3)  # Cap bonus at 0.3

        except Exception as e:
            logger.error(f"Error calculating context bonus: {str(e)}")
            return 0.0

    def extract_experience_level(self, text: str) -> Dict:
        """Extract experience level information with enhanced patterns"""
        try:
            text_lower = text.lower()
            experience_info = {
                'years': 0,
                'level': 'entry',
                'confidence': 0.0
            }

            # Extract years of experience with enhanced patterns
            max_years = 0
            for pattern in self.experience_patterns['years']:
                matches = re.findall(pattern, text_lower)
                if matches:
                    for match in matches:
                        if isinstance(match, tuple):
                            # Handle range patterns like "3 to 5 years"
                            years = max([int(m) for m in match if m.isdigit()])
                        else:
                            years = int(match) if match.isdigit() else 0
                        max_years = max(max_years, years)

            experience_info['years'] = max_years

            # Determine seniority level with scoring
            level_scores = defaultdict(int)
            for level, keywords in self.experience_patterns['seniority'].items():
                for keyword in keywords:
                    count = text_lower.count(keyword)
                    level_scores[level] += count

            # Determine best level match
            if level_scores:
                detected_level = max(level_scores.keys(), key=lambda x: level_scores[x])
                max_score = level_scores[detected_level]
                experience_info['level'] = detected_level
                experience_info['confidence'] = min(max_score * 0.2, 1.0)

            # Adjust level based on years (override if years suggest different level)
            if experience_info['years'] >= 10:
                experience_info['level'] = 'senior'
                experience_info['confidence'] = max(experience_info['confidence'], 0.8)
            elif experience_info['years'] >= 5:
                experience_info['level'] = 'mid'
                experience_info['confidence'] = max(experience_info['confidence'], 0.6)
            elif experience_info['years'] >= 2:
                experience_info['level'] = 'entry'
                experience_info['confidence'] = max(experience_info['confidence'], 0.4)

            return experience_info

        except Exception as e:
            logger.error(f"Error extracting experience level: {str(e)}")
            return {'years': 0, 'level': 'entry', 'confidence': 0.0}

    def extract_education_info(self, text: str) -> Dict:
        """Extract education information with enhanced patterns"""
        try:
            text_lower = text.lower()
            education_info = {
                'highest_degree': 'none',
                'field_of_study': [],
                'institutions': [],
                'confidence': 0.0
            }

            # Enhanced degree detection with hierarchy
            degree_hierarchy = {
                'phd': 6, 'doctorate': 6, 'doctoral': 6,
                'master': 5, 'masters': 5, 'mba': 5,
                'bachelor': 4, 'bachelors': 4,
                'associates': 3, 'associate': 3,
                'diploma': 2,
                'certificate': 1, 'certification': 1
            }

            highest_score = 0
            highest_degree = 'none'
            total_mentions = 0

            for degree, score in degree_hierarchy.items():
                pattern = rf'\b{re.escape(degree)}\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    total_mentions += matches
                    if score > highest_score:
                        highest_score = score
                        highest_degree = degree

            education_info['highest_degree'] = highest_degree
            education_info['confidence'] = min(total_mentions * 0.3, 1.0)

            # Extract fields of study with patterns
            for field in self.education_patterns['fields']:
                patterns = [
                    rf'\b{re.escape(field)}\b',
                    rf'{re.escape(field.replace(" ", ""))}',  # Without spaces
                    rf'(?:in|of)\s+{re.escape(field)}'  # "degree in field"
                ]

                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        education_info['field_of_study'].append(field)
                        break

            # Extract institutions
            education_info['institutions'] = self._extract_institutions(text)

            return education_info

        except Exception as e:
            logger.error(f"Error extracting education info: {str(e)}")
            return {'highest_degree': 'none', 'field_of_study': [], 'institutions': [], 'confidence': 0.0}

    def _extract_institutions(self, text: str) -> List[str]:
        """Extract educational institutions"""
        institutions = []

        try:
            if self.nlp:
                # Use spaCy if available
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ == "ORG" and any(
                            word in ent.text.lower()
                            for word in ['university', 'college', 'institute', 'school', 'academy']
                    ):
                        institutions.append(ent.text)
            else:
                # Fallback to regex
                institution_patterns = [
                    r'\b[A-Z][a-zA-Z\s]+University\b',
                    r'\b[A-Z][a-zA-Z\s]+College\b',
                    r'\b[A-Z][a-zA-Z\s]+Institute\b'
                ]

                for pattern in institution_patterns:
                    matches = re.findall(pattern, text)
                    institutions.extend(matches)

        except Exception as e:
            logger.error(f"Error extracting institutions: {str(e)}")

        return list(set(institutions))  # Remove duplicates