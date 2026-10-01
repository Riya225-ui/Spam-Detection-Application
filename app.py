import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer

@st.cache_resource
def download_nltk():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

download_nltk()
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    return ' '.join([ps.stem(w) for w in nltk.word_tokenize(text) if w.isalnum()])

@st.cache_resource
def load_artifacts():
    with open('vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
    return vec, clf

vectorizer, model = load_artifacts()

st.set_page_config(
    page_title="SMS / Email Spam Detector",
    page_icon="shield",
    layout="centered",
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .result-spam {
        background: linear-gradient(135deg, #ff4b4b, #ff0000);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,0,0,0.4);
    }
    .result-ham {
        background: linear-gradient(135deg, #21c55d, #15803d);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: bold;
        box-shadow: 0 4px 15px rgba(33,197,93,0.4);
    }
    .stTextArea textarea { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("## SMS / Email Spam Detector")
st.markdown("Paste any SMS or email text below and click **Analyze** to detect spam.")
st.divider()

input_text = st.text_area(
    "Enter message",
    height=160,
    placeholder="Type or paste your message here...",
)

st.markdown("")
analyze_btn = st.button("Analyze Message", type="primary", use_container_width=True)

if analyze_btn:
    if not input_text.strip():
        st.warning("Please enter a message first.")
    else:
        with st.spinner("Analyzing..."):
            processed  = transform_text(input_text)
            vec_input  = vectorizer.transform([processed]).toarray()
            prediction = model.predict(vec_input)[0]

        st.markdown("---")
        if prediction == 1:
            st.markdown('<div class="result-spam">SPAM Detected</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-ham">Not Spam (Ham)</div>',
                        unsafe_allow_html=True)
