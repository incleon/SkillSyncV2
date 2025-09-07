# SkillSync V2 🔍
### *Where Job Descriptions and Resumes Blink ;)*

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)
![NLP](https://img.shields.io/badge/NLP-spaCy-green.svg)
![License](https://img.shields.io/badge/License-Student_Project-yellow.svg)

**🌐 Live Demo:** [Click Here](https://skillsyncakt.streamlit.app/)

---

## 🎯 Overview

**SkillSync V2** is the updated and enhanced version of the original SkillSync project - an intelligent AI-powered resume matching system that helps recruiters and hiring managers find the best candidates for job positions. This version includes significant improvements in accuracy, user experience, and analytical capabilities. Using advanced Natural Language Processing (NLP) and Machine Learning techniques, it analyzes resumes against job descriptions to provide comprehensive matching scores and insights.

### 🎓 Student Project Context
This project was developed as part of a learning journey in AI/ML and demonstrates practical applications of:
- Natural Language Processing
- Machine Learning for classification
- Web application development
- Data visualization
- Semantic analysis and text mining

---

## ✨ Features

### Core Functionality
- **📄 Multi-format Support**: Upload PDF, DOCX, and TXT resume files
- **🧠 AI-Powered Matching**: Advanced semantic similarity analysis
- **📊 Comprehensive Scoring**: Multi-dimensional evaluation including:
  - Skills matching with confidence scores
  - Experience level analysis
  - Education background assessment
  - Semantic similarity scoring
  - ML-enhanced predictions

### Advanced Analytics
- **📈 Interactive Dashboards**: Real-time analytics and visualizations
- **🎯 Detailed Insights**: Skill categorization and confidence mapping
- **📊 Performance Metrics**: Score distributions and correlation analysis
- **🔍 Anomaly Detection**: Identify unusual resume patterns

### User Experience
- **🎨 Modern UI**: Clean, responsive design with gradient themes
- **⚡ Real-time Processing**: Instant analysis with progress tracking
- **📱 Mobile Friendly**: Responsive design for all devices
- **💾 Export Options**: Download results in CSV and JSON formats

---

## 🛠 Tech Stack

### **Backend & Core**
- **Python 3.8+** - Main programming language
- **Streamlit 1.28+** - Web application framework
- **NumPy & Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms

### **Natural Language Processing**
- **spaCy** - Advanced NLP with named entity recognition
- **NLTK** - Natural language processing toolkit
- **Sentence Transformers** - Semantic similarity analysis
- **TF-IDF Vectorization** - Text feature extraction

### **Machine Learning**
- **Random Forest Classifier** - Resume classification
- **Isolation Forest** - Anomaly detection
- **Standard Scaler** - Feature normalization
- **Cosine Similarity** - Text similarity measurements

### **Document Processing**
- **PyPDF2** - PDF text extraction
- **python-docx & docx2txt** - Word document processing
- **Multiple encoding support** - UTF-8, Latin-1, CP1252

### **Data Visualization**
- **Plotly** - Interactive charts and graphs
- **Matplotlib & Seaborn** - Statistical visualizations
- **WordCloud** - Skills visualization

### **Deployment**
- **Streamlit Cloud** - Hosting platform
- **Git/GitHub** - Version control
- **Requirements.txt** - Dependency management

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/SkillSyncV2.git
cd SkillSyncV2
```
### Step 1: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 4: Download spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```
### Step 5: Run the Application
```bash
streamlit run streamlit_app.py
```
The application will open in your browser at http://localhost:8501

## 🔧 How It Works

### Architecture
SkillSyncV2/
## Architecture

```bash
SkillSyncV2/
├── streamlit_app.py
├── requirements.txt
├── data/
│   └── skillsync.db
└── utils/
    ├── database_manager.py
    ├── ml_matcher.py
    ├── nlp_processor.py
    ├── semantic_matcher.py
    └── text_extractor.py
```


### Pipeline
- **Text Extraction** → `text_extractor.py`  
- **NLP Processing** → `nlp_processor.py`  
- **Semantic Matching** → `semantic_matcher.py`  
- **ML Pipeline** → `ml_matcher.py`  
- **Database Management** → `database_manager.py`  
- **Scoring** → Weighted composite scoring  
- **Analytics** → Interactive dashboards  

### Scoring Formula
```python
Composite Score = (
    Semantic Similarity × 0.25 +
    Skills Match × 0.40 +
    Experience Match × 0.20 +
    Education Match × 0.10 +
    ML Probability × 0.05
)
```

## 🚀 Future Enhancements
- [ ] User Authentication  
- [ ] AI Recommendations  
- [ ] ATS Integration  
- [ ] Multi-language Support  
- [ ] Mobile App  

---

## 🏆 Credits & Acknowledgments
**Developer**: Aditya Kumar Tiwari 
🔗 [[LinkedIn Profile](https://www.linkedin.com/in/akt11/)]   

**Thanks to**: Streamlit, spaCy, Scikit-learn, Hugging Face, Open Source  

---

## 📊 Project Stats
- 2000+ LOC  
- 85%+ accuracy (manual verification)  
- Batch: 50 resumes  
- Processing: 2–5s per resume  

---

## 🌟 Star This Project
If this helped you, ⭐ star the repo to support development!  

---

<div align="center">

🎓 Built with ❤️ for learning and education  
🚀 Deployed on Streamlit Cloud  

📈 **SkillSync V2 - The Enhanced AI Resume Matcher**  

*Happy Learning! 📚*  
*"Where Job Descriptions and Resumes Finally Sync"*  

</div>

