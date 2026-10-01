# SMS / Email Spam Detector 🛡️

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning web application built to accurately classify SMS and Email messages as **Spam** or **Not Spam (Ham)**.

### 🌐 Live Demo
You can test the application live here: **[SMS / Email Spam Detector](https://spam-detection-application-2hpkcj3gb4v53jrhyw9zzf.streamlit.app/)**

---

## 🎯 Project Goal
The primary objective of this project was to achieve **100% Precision**. In spam detection, a "False Positive" (classifying a legitimate message as spam) is highly dangerous. Therefore, this model has been strictly mathematically constrained to ensure that **no normal messages are ever marked as spam**, while still maintaining a very high recall rate.

## ⚙️ Model Architecture & Pipeline
1. **Dataset:** Trained on a dataset of 5,572 messages.
2. **Text Preprocessing:** 
   - Lowercasing, Tokenization, and Special Character Removal using `NLTK`.
   - Word Stemming using `PorterStemmer` to convert words to their root forms (e.g., "winning" -> "win").
3. **Vectorization:** 
   - Transformed the cleaned text into numerical format using `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) with a limit of 3,000 max features.
4. **Machine Learning Algorithm:** 
   - Used **Multinomial Naive Bayes (MNB)**, which is the gold standard for text classification and provides the desired 100% precision threshold.

## 📊 Performance Metrics
* **Accuracy:** 96.61%
* **Precision:** 100.00% 🎯 (Zero False Positives)
* **Recall:** 74.64%

## 💻 How to Run Locally
If you want to run this project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/Riya225-ui/Spam-Detection-Application.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
