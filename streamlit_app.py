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
import traceback

warnings.filterwarnings('ignore')

# Import your existing modules with error handling
try:
    from utils.database_manager import DatabaseManager
    from utils.nlp_processor import AdvancedNLPProcessor
    from utils.semantic_matcher import SemanticMatcher
    from utils.ml_matcher import MLEnhancedMatcher
    from utils.text_extractor import extract_text

    MODULES_LOADED = True
except ImportError as e:
    st.error(f"Error importing modules: {str(e)}")
    st.error("Please ensure all required packages are installed. Run: pip install -r requirements.txt")
    MODULES_LOADED = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .main-tagline {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
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
        background: white;
    }

    .score-high { color: #28a745; font-weight: bold; }
    .score-medium { color: #ffc107; font-weight: bold; }
    .score-low { color: #dc3545; font-weight: bold; }

    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }

    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }

    .error-box {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }

    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'job_description' not in st.session_state:
        st.session_state.job_description = ""
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"
    if 'last_analysis_time' not in st.session_state:
        st.session_state.last_analysis_time = None


class AdvancedResumeJobMatcher:
    """Enhanced matcher class for Streamlit with comprehensive error handling"""

    def __init__(self):
        try:
            self.nlp_processor = AdvancedNLPProcessor()
            self.semantic_matcher = SemanticMatcher()
            self.ml_matcher = MLEnhancedMatcher()

            # Initialize database with error handling
            db_path = os.getenv('SKILLSYNC_DB_PATH', 'data/skillsync.db')
            self.db_manager = DatabaseManager(db_path)

            logger.info("AdvancedResumeJobMatcher initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing matcher: {str(e)}")
            st.error(f"Error initializing matcher components: {str(e)}")
            raise

    def comprehensive_analysis(self, job_description: str, resume_text: str, filename: str) -> Dict:
        """Comprehensive analysis with robust error handling"""
        try:
            logger.info(f"Starting analysis for {filename}")

            # Validate inputs
            if not job_description or len(job_description.strip()) < 10:
                raise ValueError("Job description is too short or empty")

            if not resume_text or len(resume_text.strip()) < 50:
                raise ValueError("Resume text is too short or empty")

            # Extract entities and features
            try:
                job_skills = self.nlp_processor.extract_skills_advanced(job_description)
                resume_skills = self.nlp_processor.extract_skills_advanced(resume_text)
            except Exception as e:
                logger.error(f"Error in skill extraction: {str(e)}")
                job_skills = {'categorized_skills': {}, 'skill_confidence': {}, 'total_skills': 0}
                resume_skills = {'categorized_skills': {}, 'skill_confidence': {}, 'total_skills': 0}

            # Experience analysis
            try:
                resume_experience = self.nlp_processor.extract_experience_level(resume_text)
            except Exception as e:
                logger.error(f"Error in experience extraction: {str(e)}")
                resume_experience = {'years': 0, 'level': 'entry', 'confidence': 0.0}

            # Education analysis
            try:
                resume_education = self.nlp_processor.extract_education_info(resume_text)
            except Exception as e:
                logger.error(f"Error in education extraction: {str(e)}")
                resume_education = {'highest_degree': 'none', 'field_of_study': [], 'institutions': [],
                                    'confidence': 0.0}

            # Semantic similarity
            try:
                semantic_score = self.semantic_matcher.get_semantic_similarity(job_description, resume_text)
            except Exception as e:
                logger.error(f"Error in semantic analysis: {str(e)}")
                semantic_score = 0.0

            # Calculate scores
            skills_score = self._calculate_skills_match_score(job_skills, resume_skills)
            experience_score = self._calculate_experience_match_score(resume_experience)
            education_score = self._calculate_education_match_score(resume_education)

            # ML features and prediction
            try:
                ml_features = self.ml_matcher.extract_ml_features(
                    job_description, resume_text, resume_skills, resume_experience, resume_education
                )
                ml_probability = self.ml_matcher.predict_match_probability(ml_features)
            except Exception as e:
                logger.error(f"Error in ML analysis: {str(e)}")
                ml_probability = 0.0

            # Composite score
            composite_score = self._calculate_composite_score(
                semantic_score, skills_score, experience_score, education_score, ml_probability
            )

            result = {
                'filename': filename,
                'composite_score': composite_score,
                'semantic_score': semantic_score,
                'skills_score': skills_score,
                'experience_score': experience_score,
                'education_score': education_score,
                'ml_probability': ml_probability,
                'resume_skills': resume_skills,
                'resume_experience': resume_experience,
                'resume_education': resume_education,
                'analysis_success': True,
                'error_message': None
            }

            logger.info(f"Analysis completed successfully for {filename}")
            return result

        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {filename}: {str(e)}")
            return self._get_fallback_analysis(filename, str(e))

    def _calculate_skills_match_score(self, job_skills: Dict, resume_skills: Dict) -> float:
        """Calculate skills matching score with error handling"""
        try:
            job_skill_set = set(job_skills.get('skill_confidence', {}).keys())
            resume_skill_set = set(resume_skills.get('skill_confidence', {}).keys())

            if not job_skill_set:
                return 0.0

            common_skills = job_skill_set & resume_skill_set
            if not common_skills:
                return 0.0

            # Weight by confidence scores
            total_weight = 0
            matched_weight = 0

            for skill in job_skill_set:
                weight = job_skills.get('skill_confidence', {}).get(skill, 0.5)
                total_weight += weight

                if skill in common_skills:
                    resume_confidence = resume_skills.get('skill_confidence', {}).get(skill, 0.5)
                    matched_weight += min(weight, resume_confidence)

            return matched_weight / total_weight if total_weight > 0 else 0.0

        except Exception as e:
            logger.error(f"Error calculating skills match: {str(e)}")
            return 0.0

    def _calculate_experience_match_score(self, resume_exp: Dict) -> float:
        """Calculate experience score with enhanced logic"""
        try:
            years = resume_exp.get('years', 0)
            level = resume_exp.get('level', 'entry')
            confidence = resume_exp.get('confidence', 0.0)

            # Level-based scoring
            level_scores = {
                'entry': 0.3,
                'mid': 0.6,
                'senior': 0.9,
                'executive': 1.0
            }

            # Years-based scoring (logarithmic scale)
            years_score = min(np.log10(max(years, 1)) / np.log10(20), 1.0)
            level_score = level_scores.get(level, 0.3)

            # Combine scores with confidence weighting
            combined_score = (years_score * 0.6 + level_score * 0.4)
            return combined_score * (0.5 + confidence * 0.5)

        except Exception as e:
            logger.error(f"Error calculating experience score: {str(e)}")
            return 0.0

    def _calculate_education_match_score(self, resume_education: Dict) -> float:
        """Calculate education score with field matching"""
        try:
            degree_scores = {
                'phd': 1.0, 'doctorate': 1.0, 'doctoral': 1.0,
                'master': 0.8, 'masters': 0.8, 'mba': 0.8,
                'bachelor': 0.6, 'bachelors': 0.6,
                'associates': 0.4, 'associate': 0.4,
                'diploma': 0.3,
                'certificate': 0.2, 'certification': 0.2,
                'none': 0.0
            }

            degree = resume_education.get('highest_degree', 'none').lower()
            base_score = degree_scores.get(degree, 0.0)

            # Bonus for relevant field of study
            relevant_fields = ['computer science', 'engineering', 'technology']
            field_bonus = 0.1 if any(
                field in resume_education.get('field_of_study', [])
                for field in relevant_fields
            ) else 0.0

            confidence = resume_education.get('confidence', 0.5)

            return min((base_score + field_bonus) * (0.7 + confidence * 0.3), 1.0)

        except Exception as e:
            logger.error(f"Error calculating education score: {str(e)}")
            return 0.0

    def _calculate_composite_score(self, semantic_score: float, skills_score: float,
                                   experience_score: float, education_score: float,
                                   ml_probability: float) -> float:
        """Calculate weighted composite score"""
        try:
            # Dynamic weights based on score reliability
            weights = {
                'semantic': 0.25,
                'skills': 0.40,
                'experience': 0.20,
                'education': 0.10,
                'ml': 0.05
            }

            # Ensure all scores are valid
            scores = {
                'semantic': max(0.0, min(semantic_score, 1.0)),
                'skills': max(0.0, min(skills_score, 1.0)),
                'experience': max(0.0, min(experience_score, 1.0)),
                'education': max(0.0, min(education_score, 1.0)),
                'ml': max(0.0, min(ml_probability, 1.0))
            }

            composite = sum(scores[key] * weights[key] for key in weights.keys())
            return max(0.0, min(composite, 1.0))

        except Exception as e:
            logger.error(f"Error calculating composite score: {str(e)}")
            return 0.0

    def _get_fallback_analysis(self, filename: str, error_message: str = None) -> Dict:
        """Fallback analysis result with error information"""
        return {
            'filename': filename,
            'composite_score': 0.0,
            'semantic_score': 0.0,
            'skills_score': 0.0,
            'experience_score': 0.0,
            'education_score': 0.0,
            'ml_probability': 0.0,
            'resume_skills': {'categorized_skills': {}, 'skill_confidence': {}, 'total_skills': 0},
            'resume_experience': {'years': 0, 'level': 'entry', 'confidence': 0.0},
            'resume_education': {'highest_degree': 'none', 'field_of_study': [], 'institutions': [], 'confidence': 0.0},
            'analysis_success': False,
            'error_message': error_message,
            'success': False
        }


@st.cache_resource
def get_matcher():
    """Initialize matcher with caching"""
    if not MODULES_LOADED:
        return None
    try:
        return AdvancedResumeJobMatcher()
    except Exception as e:
        st.error(f"Failed to initialize matcher: {str(e)}")
        return None


def display_system_status():
    """Display system status and health checks"""
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            # Modules status
            if MODULES_LOADED:
                st.success("✅ Core modules loaded")
            else:
                st.error("❌ Module loading failed")

        with col2:
            # Matcher status
            matcher = get_matcher()
            if matcher:
                st.success("✅ Matcher initialized")
            else:
                st.error("❌ Matcher failed")

        with col3:
            # NLP status
            try:
                from utils.nlp_processor import SPACY_AVAILABLE
                if SPACY_AVAILABLE:
                    st.success("✅ Advanced NLP available")
                else:
                    st.warning("⚠️ Basic NLP mode")
            except:
                st.error("❌ NLP unavailable")


def validate_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file"""
    try:
        # Check file size
        max_size = 10 * 1024 * 1024  # 10MB
        if uploaded_file.size > max_size:
            return False, f"File too large: {uploaded_file.size / (1024 * 1024):.1f}MB (max: 10MB)"

        # Check file extension
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        if file_ext not in allowed_extensions:
            return False, f"Unsupported file type: {file_ext}"

        # Check filename
        if len(uploaded_file.name) > 100:
            return False, "Filename too long"

        return True, "File is valid"

    except Exception as e:
        return False, f"File validation error: {str(e)}"


def process_single_resume(job_description: str, uploaded_file, matcher) -> Dict:
    """Process a single resume with comprehensive error handling"""
    try:
        # Validate file first
        is_valid, validation_message = validate_uploaded_file(uploaded_file)
        if not is_valid:
            return {
                'filename': uploaded_file.name,
                'error': validation_message,
                'success': False
            }

        # Create temporary file
        with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=f".{uploaded_file.name.split('.')[-1]}"
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            # Extract text
            resume_text = extract_text(tmp_file_path)

            if not resume_text or len(resume_text.strip()) < 50:
                return {
                    'filename': uploaded_file.name,
                    'error': 'Could not extract sufficient text from file',
                    'success': False
                }

            # Perform analysis
            result = matcher.comprehensive_analysis(
                job_description, resume_text, uploaded_file.name
            )
            result['success'] = True
            return result

        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass

    except Exception as e:
        logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return {
            'filename': uploaded_file.name,
            'error': str(e),
            'success': False
        }


def display_detailed_results(results: List[Dict]):
    """Display detailed analysis results with proper formatting"""
    if not results:
        st.info("No results to display")
        return

    # Summary statistics
    st.subheader("📊 Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    successful_results = [r for r in results if r.get('success', False)]
    failed_results = [r for r in results if not r.get('success', False)]

    with col1:
        st.metric("Total Resumes", len(results))
    with col2:
        st.metric("Successfully Analyzed", len(successful_results))
    with col3:
        st.metric("Failed Analysis", len(failed_results))
    with col4:
        if successful_results:
            avg_score = np.mean([r['composite_score'] for r in successful_results]) * 100
            st.metric("Average Match", f"{avg_score:.1f}%")
        else:
            st.metric("Average Match", "N/A")

    # Show failed analyses if any
    if failed_results:
        st.subheader("⚠️ Failed Analyses")
        for result in failed_results:
            st.error(f"**{result['filename']}**: {result.get('error', 'Unknown error')}")

    # Display successful results
    if successful_results:
        st.subheader("🎯 Matching Results")

        # Sort by composite score
        successful_results.sort(key=lambda x: x['composite_score'], reverse=True)

        for i, result in enumerate(successful_results):
            score = result['composite_score'] * 100

            # Determine score status
            if score >= 80:
                score_color = "#28a745"  # Green
                score_emoji = "🟢"
                score_bg = "#d4edda"
            elif score >= 60:
                score_color = "#ffc107"  # Yellow
                score_emoji = "🟡"
                score_bg = "#fff3cd"
            else:
                score_color = "#dc3545"  # Red
                score_emoji = "🔴"
                score_bg = "#f8d7da"

            # Create result card using native Streamlit components
            with st.container():
                # Card styling
                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e0e0;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    background: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                        <h4 style="margin: 0; color: #333;">{score_emoji} #{i + 1} {result['filename']}</h4>
                        <div style="
                            background: {score_bg};
                            color: {score_color};
                            padding: 8px 15px;
                            border-radius: 20px;
                            font-weight: bold;
                            font-size: 16px;
                        ">
                            Overall: {score:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Score breakdown using Streamlit columns
                st.markdown("**📊 Score Breakdown:**")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    skills_score = result['skills_score'] * 100
                    st.metric("🛠️ Skills", f"{skills_score:.1f}%")

                with col2:
                    exp_score = result['experience_score'] * 100
                    st.metric("👔 Experience", f"{exp_score:.1f}%")

                with col3:
                    edu_score = result['education_score'] * 100
                    st.metric("🎓 Education", f"{edu_score:.1f}%")

                with col4:
                    sem_score = result['semantic_score'] * 100
                    st.metric("🧠 Semantic", f"{sem_score:.1f}%")

                # Expandable details
                with st.expander(f"📋 Detailed Analysis - {result['filename']}", expanded=False):
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.markdown("### 📈 Experience Profile")
                        exp = result['resume_experience']

                        # Experience info in a nice format
                        st.info(f"""
                        **Years of Experience:** {exp.get('years', 0)} years  
                        **Seniority Level:** {exp.get('level', 'Unknown').title()}  
                        **Confidence Score:** {exp.get('confidence', 0) * 100:.1f}%
                        """)

                        st.markdown("### 🎓 Educational Background")
                        edu = result['resume_education']

                        degree = edu.get('highest_degree', 'None').title()
                        fields = edu.get('field_of_study', [])
                        institutions = edu.get('institutions', [])

                        edu_info = f"**Highest Degree:** {degree}\n"
                        if fields:
                            edu_info += f"**Field(s) of Study:** {', '.join(fields)}\n"
                        if institutions:
                            edu_info += f"**Institution(s):** {', '.join(institutions[:2])}"

                        st.info(edu_info)

                    with detail_col2:
                        st.markdown("### 🛠️ Skills Analysis")
                        skills = result['resume_skills']

                        # Skills summary
                        total_skills = skills.get('total_skills', 0)
                        st.metric("Total Skills Detected", total_skills)

                        # Categorized skills
                        categorized_skills = skills.get('categorized_skills', {})
                        if categorized_skills:
                            st.markdown("**Skills by Category:**")
                            for category, skill_list in list(categorized_skills.items())[:4]:  # Show max 4 categories
                                if skill_list:
                                    category_name = category.replace('_', ' ').title()
                                    skills_text = ', '.join(
                                        [s['skill'] for s in skill_list[:4]])  # Max 4 skills per category
                                    st.write(f"• **{category_name}**: {skills_text}")

                        # Top skills by confidence
                        skill_confidence = skills.get('skill_confidence', {})
                        if skill_confidence:
                            st.markdown("**Top Skills by Confidence:**")
                            top_skills = sorted(skill_confidence.items(), key=lambda x: x[1], reverse=True)[:6]
                            for skill, confidence in top_skills:
                                confidence_percent = confidence * 100
                                st.write(f"• {skill}: {confidence_percent:.0f}%")

                        # ML insights
                        if 'ml_probability' in result:
                            ml_prob = result['ml_probability'] * 100
                            st.metric("🤖 ML Match Probability", f"{ml_prob:.1f}%")

                # Add visual separator
                st.markdown("---")


def main():
    """Main application function with comprehensive error handling"""
    try:
        initialize_session_state()

        # Header
        st.markdown('<h1 class="main-header">SkillSync V2</h1>', unsafe_allow_html=True)
        st.markdown('<p class="main-tagline">Where JDs and Resumes Blink</p>', unsafe_allow_html=True)

        # System status
        display_system_status()

        # Check if system is ready
        if not MODULES_LOADED:
            st.error("System is not ready. Please check the setup instructions.")
            st.markdown("""
            <div class="error-box">
                <h4>Setup Required</h4>
                <p>Please install all required packages:</p>
                <code>pip install -r requirements.txt</code>
                <br><br>
                <p>For enhanced NLP features, also run:</p>
                <code>python -m spacy download en_core_web_sm</code>
            </div>
            """, unsafe_allow_html=True)
            return

        # Sidebar configuration
        st.sidebar.title("⚙️ Configuration")

        # Analysis settings
        st.sidebar.subheader("Analysis Settings")
        max_results = st.sidebar.slider("Max Results to Show", 5, 50, 20)
        show_detailed_analysis = st.sidebar.checkbox("Show Detailed Analysis", value=True)

        # Performance settings
        st.sidebar.subheader("Performance Settings")
        batch_size = st.sidebar.slider("Batch Processing Size", 1, 10, 5)

        # Get matcher instance
        matcher = get_matcher()
        if not matcher:
            st.error("Failed to initialize the matching system. Please check the logs.")
            return

        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["🎯 Match Resumes", "📈 Results", "📊 Analytics"])

        with tab1:
            st.subheader("Job Description & Resume Upload")

            # Job description input
            job_description = st.text_area(
                "Enter Job Description",
                height=200,
                placeholder="Paste the complete job description here...\n\nInclude:\n- Required skills and technologies\n- Experience requirements\n- Educational qualifications\n- Job responsibilities",
                help="Enter a detailed job description with required skills, experience, and qualifications for better matching accuracy."
            )

            # Resume upload
            uploaded_files = st.file_uploader(
                "Upload Resume Files",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload multiple resume files in PDF, DOCX, or TXT format. Maximum file size: 10MB each."
            )

            # File validation info
            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} file(s) selected")

                # Show file details
                with st.expander("📋 File Details"):
                    for file in uploaded_files:
                        file_size = file.size / (1024 * 1024)  # MB
                        st.write(f"• **{file.name}** ({file_size:.1f} MB)")

            # Analysis button
            col1, col2 = st.columns([3, 1])
            with col1:
                analyze_button = st.button("🚀 Analyze Resumes", type="primary", use_container_width=True)
            with col2:
                clear_button = st.button("🗑️ Clear Results", use_container_width=True)

            if clear_button:
                st.session_state.analysis_results = []
                st.session_state.last_analysis_time = None
                st.success("Results cleared!")
                st.rerun()

            # Analysis execution
            if analyze_button:
                # Validation
                if not job_description or len(job_description.strip()) < 50:
                    st.error("⚠️ Please enter a detailed job description (at least 50 characters).")
                elif not uploaded_files:
                    st.error("⚠️ Please upload at least one resume file.")
                else:
                    # Process resumes
                    st.session_state.job_description = job_description
                    st.session_state.processing_status = "processing"

                    results = []

                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    try:
                        # Process files in batches
                        total_files = len(uploaded_files)

                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f'Processing {uploaded_file.name}... ({i + 1}/{total_files})')

                            # Process single resume
                            result = process_single_resume(job_description, uploaded_file, matcher)
                            results.append(result)

                            # Update progress
                            progress_bar.progress((i + 1) / total_files)

                            # Small delay for UX
                            if i < total_files - 1:
                                import time
                                time.sleep(0.1)

                        # Sort results by composite score (successful ones first)
                        successful_results = [r for r in results if r.get('success', False)]
                        failed_results = [r for r in results if not r.get('success', False)]

                        successful_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
                        results = successful_results + failed_results

                        # Store results
                        st.session_state.analysis_results = results
                        st.session_state.last_analysis_time = datetime.now()
                        st.session_state.processing_status = "completed"

                        # Show completion status
                        status_text.empty()
                        progress_bar.empty()

                        success_count = len(successful_results)
                        fail_count = len(failed_results)

                        if success_count > 0:
                            st.success(f"✅ Analysis completed! {success_count} resumes analyzed successfully.")
                            if fail_count > 0:
                                st.warning(f"⚠️ {fail_count} files could not be processed.")
                        else:
                            st.error("❌ No resumes could be processed. Please check file formats and content.")

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        logger.error(f"Analysis error: {str(e)}\n{traceback.format_exc()}")
                        st.session_state.processing_status = "error"

        with tab2:
            st.subheader("🎯 Analysis Results")

            if st.session_state.analysis_results:
                results = st.session_state.analysis_results[:max_results]

                # Analysis timestamp
                if st.session_state.last_analysis_time:
                    st.caption(f"📅 Last analysis: {st.session_state.last_analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")

                # Display results using the fixed function
                if show_detailed_analysis:
                    display_detailed_results(results)

                # Results table
                successful_results = [r for r in results if r.get('success', False)]
                if successful_results:
                    st.subheader("📋 Summary Table")

                    df_results = pd.DataFrame([{
                        'Rank': i + 1,
                        'Filename': r['filename'],
                        'Overall Score': f"{r['composite_score'] * 100:.1f}%",
                        'Skills Match': f"{r['skills_score'] * 100:.1f}%",
                        'Experience': f"{r['experience_score'] * 100:.1f}%",
                        'Education': f"{r['education_score'] * 100:.1f}%",
                        'Semantic': f"{r['semantic_score'] * 100:.1f}%",
                    } for i, r in enumerate(successful_results)])

                    st.dataframe(df_results, use_container_width=True)

                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"skillsync_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with col2:
                        # JSON download with full results
                        json_data = json.dumps(successful_results, indent=2, default=str)
                        st.download_button(
                            label="📥 Download Full Data (JSON)",
                            data=json_data,
                            file_name=f"skillsync_full_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )

            else:
                st.info("💡 No results available. Please run the analysis first in the 'Match Resumes' tab.")

                # Quick start guide with better formatting
                with st.container():
                    st.markdown("""
                    ### 🚀 Quick Start Guide

                    Follow these steps to analyze resumes:

                    1. **📝 Go to the 'Match Resumes' tab**
                    2. **📄 Enter a detailed job description** (minimum 50 characters)
                       - Include specific skills and technologies
                       - Mention experience requirements
                       - Add educational qualifications
                    3. **📁 Upload resume files** (PDF, DOCX, or TXT format)
                    4. **🚀 Click 'Analyze Resumes'** to start matching
                    5. **📊 View results here** once analysis completes

                    ### 💡 Tips for Better Results:
                    - Use detailed job descriptions with specific requirements
                    - Upload well-formatted, readable resume files
                    - Include both technical and soft skills in job descriptions
                    - Ensure resume files are not corrupted or password-protected
                    """)

                    # Sample job description
                    with st.expander("📋 Sample Job Description"):
                        st.code("""
Sample Job Description:

We are looking for a Senior Python Developer with 5+ years of experience.

Required Skills:
- Python, Django, Flask
- JavaScript, React, Node.js
- PostgreSQL, MongoDB
- AWS, Docker, Kubernetes
- Git, CI/CD pipelines

Experience:
- 5+ years in software development
- Experience with microservices architecture
- API design and development
- Agile/Scrum methodology

Education:
- Bachelor's degree in Computer Science or related field
- Master's degree preferred

Responsibilities:
- Lead development of web applications
- Mentor junior developers
- Code review and quality assurance
- Collaborate with cross-functional teams
                        """)

        with tab3:
            st.subheader("📊 Analytics Dashboard")

            if st.session_state.analysis_results:
                successful_results = [r for r in st.session_state.analysis_results if r.get('success', False)]

                if successful_results:
                    # Score distribution
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("🎯 Score Distribution")
                        scores = [r['composite_score'] * 100 for r in successful_results]

                        fig = px.histogram(
                            x=scores,
                            nbins=10,
                            title="Distribution of Match Scores",
                            labels={'x': 'Match Score (%)', 'y': 'Number of Resumes'}
                        )
                        fig.update_layout(
                            xaxis_title="Match Score (%)",
                            yaxis_title="Number of Resumes",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.subheader("🏆 Top Performers")

                        # Top 5 candidates
                        top_candidates = successful_results[:5]

                        fig = px.bar(
                            x=[r['filename'][:20] + '...' if len(r['filename']) > 20 else r['filename'] for r in
                               top_candidates],
                            y=[r['composite_score'] * 100 for r in top_candidates],
                            title="Top 5 Candidates",
                            labels={'x': 'Resume', 'y': 'Match Score (%)'}
                        )
                        fig.update_layout(
                            xaxis_title="Resume Files",
                            yaxis_title="Match Score (%)",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Skills analysis
                    st.subheader("🛠️ Skills Analysis")

                    # Collect all skills
                    all_skills = {}
                    for result in successful_results:
                        skills = result['resume_skills'].get('skill_confidence', {})
                        for skill, confidence in skills.items():
                            if skill in all_skills:
                                all_skills[skill].append(confidence)
                            else:
                                all_skills[skill] = [confidence]

                    # Most common skills
                    skill_counts = {skill: len(confidences) for skill, confidences in all_skills.items()}
                    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:15]

                    if top_skills:
                        col1, col2 = st.columns(2)

                        with col1:
                            fig = px.bar(
                                x=[skill for skill, count in top_skills],
                                y=[count for skill, count in top_skills],
                                title="Most Common Skills",
                                labels={'x': 'Skills', 'y': 'Number of Resumes'}
                            )
                            fig.update_layout(
                                xaxis_title="Skills",
                                yaxis_title="Frequency",
                                xaxis={'tickangle': 45}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            # Average skill confidence
                            skill_avg_confidence = {
                                skill: np.mean(confidences)
                                for skill, confidences in all_skills.items()
                            }
                            top_confidence_skills = sorted(
                                skill_avg_confidence.items(),
                                key=lambda x: x[1],
                                reverse=True
                            )[:10]

                            fig = px.bar(
                                x=[skill for skill, conf in top_confidence_skills],
                                y=[conf for skill, conf in top_confidence_skills],
                                title="Skills by Average Confidence",
                                labels={'x': 'Skills', 'y': 'Average Confidence'}
                            )
                            fig.update_layout(
                                xaxis_title="Skills",
                                yaxis_title="Average Confidence",
                                xaxis={'tickangle': 45}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    # Experience level distribution
                    st.subheader("👔 Experience Level Distribution")

                    experience_levels = [r['resume_experience'].get('level', 'unknown') for r in successful_results]
                    level_counts = pd.Series(experience_levels).value_counts()

                    col1, col2 = st.columns(2)

                    with col1:
                        fig = px.pie(
                            values=level_counts.values,
                            names=level_counts.index,
                            title="Experience Levels"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Years of experience distribution
                        years_exp = [r['resume_experience'].get('years', 0) for r in successful_results]

                        fig = px.box(
                            y=years_exp,
                            title="Years of Experience Distribution"
                        )
                        fig.update_layout(
                            yaxis_title="Years of Experience"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Education analysis
                    st.subheader("🎓 Education Analysis")

                    education_levels = [r['resume_education'].get('highest_degree', 'none') for r in successful_results]
                    edu_counts = pd.Series(education_levels).value_counts()

                    fig = px.bar(
                        x=edu_counts.index,
                        y=edu_counts.values,
                        title="Education Level Distribution",
                        labels={'x': 'Education Level', 'y': 'Number of Candidates'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Score correlation matrix
                    st.subheader("🔗 Score Correlations")

                    score_data = pd.DataFrame([{
                        'Composite': r['composite_score'],
                        'Skills': r['skills_score'],
                        'Experience': r['experience_score'],
                        'Education': r['education_score'],
                        'Semantic': r['semantic_score']
                    } for r in successful_results])

                    correlation_matrix = score_data.corr()

                    fig = px.imshow(
                        correlation_matrix.values,
                        x=correlation_matrix.columns,
                        y=correlation_matrix.index,
                        color_continuous_scale='RdBu',
                        title="Score Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Performance insights
                    st.subheader("💡 Insights")

                    # Calculate insights
                    avg_composite = np.mean([r['composite_score'] for r in successful_results]) * 100
                    avg_skills = np.mean([r['skills_score'] for r in successful_results]) * 100
                    avg_experience = np.mean([r['experience_score'] for r in successful_results]) * 100

                    high_performers = len([r for r in successful_results if r['composite_score'] >= 0.8])
                    medium_performers = len([r for r in successful_results if 0.6 <= r['composite_score'] < 0.8])
                    low_performers = len([r for r in successful_results if r['composite_score'] < 0.6])

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"""
                        **📈 Performance Summary:**
                        - Average match score: {avg_composite:.1f}%
                        - Average skills match: {avg_skills:.1f}%
                        - Average experience match: {avg_experience:.1f}%
                        - High performers (≥80%): {high_performers} candidates
                        - Medium performers (60-79%): {medium_performers} candidates
                        - Needs improvement (<60%): {low_performers} candidates
                        """)

                    with col2:
                        st.markdown(f"""
                        **🎯 Recommendations:**
                        - {'Focus on skills development programs' if avg_skills < 50 else 'Strong skills alignment observed'}
                        - {'Consider broader experience criteria' if avg_experience < 40 else 'Good experience match'}
                        - {'Review job requirements' if avg_composite < 30 else 'Good candidate pool quality'}
                        - {'Expand candidate sourcing' if high_performers < 3 else 'Sufficient high-quality candidates'}
                        """)

                else:
                    st.info("📊 No successful analyses available for analytics.")

            else:
                st.info("📊 No data available for analytics. Run some analyses first!")

    except Exception as e:
        logger.error(f"Main application error: {str(e)}\n{traceback.format_exc()}")
        st.error(f"❌ Application error: {str(e)}")

        # Show error details in expander for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("💥 Critical application error occurred!")
        st.exception(e)