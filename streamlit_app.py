import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Import your existing modules
from utils.database_manager import DatabaseManager
from utils.nlp_processor import AdvancedNLPProcessor
from utils.semantic_matcher import SemanticMatcher
from utils.ml_matcher import MLEnhancedMatcher
from utils.text_extractor import extract_text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SkillSync - AI Resume Matcher",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #12063b 0%, #09555c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin: 0.5rem 0;
    }

    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def initialize_session_state():
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""


class AdvancedResumeJobMatcher:
    """Enhanced matcher class for Streamlit"""

    def __init__(self):
        self.nlp_processor = AdvancedNLPProcessor()
        self.semantic_matcher = SemanticMatcher()
        self.ml_matcher = MLEnhancedMatcher()
        self.db_manager = DatabaseManager('data/skillsync.db')

    def comprehensive_analysis(self, job_description: str, resume_text: str, filename: str) -> Dict:
        """Comprehensive analysis without caching"""
        try:
            # Extract entities and features
            job_skills = self.nlp_processor.extract_skills_advanced(job_description)
            resume_skills = self.nlp_processor.extract_skills_advanced(resume_text)

            # Experience analysis
            resume_experience = self.nlp_processor.extract_experience_level(resume_text)

            # Education analysis
            resume_education = self.nlp_processor.extract_education_info(resume_text)

            # Semantic similarity
            semantic_score = self.semantic_matcher.get_semantic_similarity(job_description, resume_text)

            # Calculate scores
            skills_score = self._calculate_skills_match_score(job_skills, resume_skills)
            experience_score = self._calculate_experience_match_score(resume_experience)
            education_score = self._calculate_education_match_score(resume_education)

            # ML features and prediction
            ml_features = self.ml_matcher.extract_ml_features(
                job_description, resume_text, resume_skills, resume_experience, resume_education
            )
            ml_probability = self.ml_matcher.predict_match_probability(ml_features)

            # Composite score
            composite_score = self._calculate_composite_score(
                semantic_score, skills_score, experience_score, education_score, ml_probability
            )

            return {
                'filename': filename,
                'composite_score': composite_score,
                'semantic_score': semantic_score,
                'skills_score': skills_score,
                'experience_score': experience_score,
                'education_score': education_score,
                'ml_probability': ml_probability,
                'resume_skills': resume_skills,
                'resume_experience': resume_experience,
                'resume_education': resume_education
            }

        except Exception as e:
            logger.error(f"Error in analysis for {filename}: {str(e)}")
            return self._get_fallback_analysis(filename)

    def _calculate_skills_match_score(self, job_skills: Dict, resume_skills: Dict) -> float:
        """Calculate skills matching score"""
        try:
            job_skill_set = set(job_skills.get('skill_confidence', {}).keys())
            resume_skill_set = set(resume_skills.get('skill_confidence', {}).keys())

            if not job_skill_set:
                return 0.0

            common_skills = job_skill_set & resume_skill_set
            if not common_skills:
                return 0.0

            return len(common_skills) / len(job_skill_set)

        except Exception as e:
            logger.error(f"Error calculating skills match: {str(e)}")
            return 0.0

    def _calculate_experience_match_score(self, resume_exp: Dict) -> float:
        """Calculate experience score"""
        try:
            years = resume_exp.get('years', 0)
            level = resume_exp.get('level', 'entry')

            level_scores = {'entry': 0.3, 'mid': 0.6, 'senior': 0.9, 'executive': 1.0}
            years_score = min(years / 10, 1.0)
            level_score = level_scores.get(level, 0.3)

            return (years_score * 0.6 + level_score * 0.4)

        except Exception as e:
            logger.error(f"Error calculating experience score: {str(e)}")
            return 0.0

    def _calculate_education_match_score(self, resume_education: Dict) -> float:
        """Calculate education score"""
        try:
            degree_scores = {
                'phd': 1.0, 'doctorate': 1.0,
                'master': 0.8, 'bachelor': 0.6,
                'associates': 0.4, 'diploma': 0.3,
                'certificate': 0.2, 'none': 0.0
            }

            degree = resume_education.get('highest_degree', 'none')
            return degree_scores.get(degree, 0.0)

        except Exception as e:
            logger.error(f"Error calculating education score: {str(e)}")
            return 0.0

    def _calculate_composite_score(self, semantic_score: float, skills_score: float,
                                   experience_score: float, education_score: float, ml_probability: float) -> float:
        """Calculate weighted composite score"""
        weights = {'semantic': 0.3, 'skills': 0.35, 'experience': 0.2, 'education': 0.1, 'ml': 0.05}

        composite = (
                semantic_score * weights['semantic'] +
                skills_score * weights['skills'] +
                experience_score * weights['experience'] +
                education_score * weights['education'] +
                ml_probability * weights['ml']
        )

        return min(max(composite, 0.0), 1.0)

    def _get_fallback_analysis(self, filename: str) -> Dict:
        """Fallback analysis result"""
        return {
            'filename': filename,
            'composite_score': 0.0,
            'semantic_score': 0.0,
            'skills_score': 0.0,
            'experience_score': 0.0,
            'education_score': 0.0,
            'ml_probability': 0.0,
            'resume_skills': {},
            'resume_experience': {},
            'resume_education': {}
        }


# Initialize matcher without caching for now
def get_matcher():
    return AdvancedResumeJobMatcher()


def main():
    """Main application function"""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🔍 SkillSync - AI Resume Matcher</h1>', unsafe_allow_html=True)
    st.markdown("### *Where Job Descriptions and Resumes Click!*")

    # Sidebar
    st.sidebar.title("⚙️ Configuration")

    # Analysis settings
    st.sidebar.subheader("Analysis Settings")
    max_results = st.sidebar.slider("Max Results to Show", 5, 50, 20)

    # Get matcher instance
    matcher = get_matcher()

    # Main content
    tab1, tab2 = st.tabs(["🎯 Match Resumes", "📈 Results"])

    with tab1:
        st.subheader("Job Description & Resume Upload")

        # Job description input
        job_description = st.text_area(
            "Enter Job Description",
            height=200,
            placeholder="Paste the complete job description here...",
            help="Enter a detailed job description with required skills, experience, and qualifications."
        )

        # Resume upload
        uploaded_files = st.file_uploader(
            "Upload Resume Files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload multiple resume files in PDF, DOCX, or TXT format."
        )

        # Analysis button
        if st.button("🚀 Analyze Resumes", type="primary"):
            if not job_description or len(job_description.strip()) < 50:
                st.error("Please enter a detailed job description (at least 50 characters).")
            elif not uploaded_files:
                st.error("Please upload at least one resume file.")
            else:
                # Process resumes
                st.session_state.job_description = job_description
                results = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f'Processing {uploaded_file.name}...')

                    # Save temporary file
                    with tempfile.NamedTemporaryFile(delete=False,
                                                     suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name

                    try:
                        # Extract text
                        resume_text = extract_text(tmp_file_path)

                        if resume_text and len(resume_text.strip()) > 50:
                            # Perform analysis
                            result = matcher.comprehensive_analysis(
                                job_description, resume_text, uploaded_file.name
                            )
                            results.append(result)
                        else:
                            st.warning(f"Could not extract sufficient content from {uploaded_file.name}")

                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")

                    finally:
                        # Clean up
                        try:
                            os.unlink(tmp_file_path)
                        except:
                            pass

                    progress_bar.progress((i + 1) / len(uploaded_files))

                # Sort results by composite score
                results.sort(key=lambda x: x['composite_score'], reverse=True)
                st.session_state.analysis_results = results

                status_text.text("Analysis complete!")
                st.success(f"✅ Successfully analyzed {len(results)} resumes!")

    with tab2:
        st.subheader("🎯 Matching Results")

        if st.session_state.analysis_results:
            results = st.session_state.analysis_results[:max_results]

            # Top candidates summary
            st.markdown("### 🏆 Top Candidates")

            for i, result in enumerate(results):
                with st.container():
                    score = result['composite_score'] * 100
                    if score >= 80:
                        score_class = "score-high"
                    elif score >= 60:
                        score_class = "score-medium"
                    else:
                        score_class = "score-low"

                    st.markdown(f"""
                    <div class="result-card">
                        <h4>#{i + 1} {result['filename']}</h4>
                        <p class="{score_class}">Overall Match: {score:.1f}%</p>
                        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                            <span>Skills: {result['skills_score'] * 100:.1f}%</span>
                            <span>Experience: {result['experience_score'] * 100:.1f}%</span>
                            <span>Education: {result['education_score'] * 100:.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Detailed results table
            st.markdown("### 📋 Detailed Results")

            df_results = pd.DataFrame([{
                'Rank': i + 1,
                'Filename': r['filename'],
                'Overall Score': f"{r['composite_score'] * 100:.1f}%",
                'Skills Match': f"{r['skills_score'] * 100:.1f}%",
                'Experience': f"{r['experience_score'] * 100:.1f}%",
                'Education': f"{r['education_score'] * 100:.1f}%",
                'Semantic': f"{r['semantic_score'] * 100:.1f}%",
            } for i, r in enumerate(results)])

            st.dataframe(df_results, use_container_width=True)

            # Download results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"skillsync_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        else:
            st.info("No results available. Please run the analysis first in the 'Match Resumes' tab.")


if __name__ == "__main__":
    main()