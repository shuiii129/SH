# Project Models
The model files were too large for GitHub, so you can download them here:
- [Download Model (.keras)](https://drive.google.com/file/d/1Be-07y6zoxu4k3AaTtLhoLgdjuxM_tq_/view?usp=sharing)
- [Download Scaler (.pkl)](https://drive.google.com/file/d/1LwUPL2kog_2JBfbFczsulCrGTtJB_BQK/view?usp=sharing)


AmanPay AI - Explainable Fraud Detection
Protecting Malaysian from digital payment fraud using AI

# The Problem
-Inadequacy of Rule-Based Systems
Traditional banking system relied on rule-based systems, this will lead to the system fail to detect unknown fraud pattern and incorrectly block legitimate transaction from rural users.
-Digital Exclusion Driven by Security Opacity
Current fraud system act as “Black Box”, block transaction without explanation, this might lead to the confusion of the users.
-Absence of Real-Time Transactional Visibility
Users often lack visibility into their transaction’s security status. When real time transactions are happening, users feel lack of control and distrust banking infrastructure.


# Solution
AmanPay AI detects fraud AND explains why in Malaysian language:
-TensorFlow Autoencoder - Learns your spending patterns
-Google Gemini AI - Explains risks in Malaysian slang
-Firebase Database - Transparent real-time logging


# Quick Start
1. Install Dependencies
bashpip install tensorflow google-generativeai firebase-admin streamlit folium streamlit-folium pandas numpy scikit-learn joblib
2. Download Pre-trained Model

Model (.keras)
Scaler (.pkl)

Place files in project folder.
3. Add Your API Key
Edit amanpay_ai_app.py line 77:
python 
GEMINI_API_KEY = "your-key-here"  # Get free at https://ai.google.dev/
4. Run
bashstreamlit run app.py


# Tech Stack
Machine Learning- TensorFlow Autoencoder (deep learning)
Artificial Intelligence- Google Gemini 1.5 Flash (explainable)
Database- Firebase Realtime Database
Frontend- Streamlit and Google Maps


# Impact
SDG 8 Decent Work and Economic Growth 
Target 8.10: Strengthen the capacity of domestic financial institutions to encourage and expand access to banking for all.


SDG 10 Reduced Inequalities
Target 10.2: Empower and promote the social, economic and political inclusion of all. We bridge the technology gap for non-technical users.



