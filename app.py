import streamlit as st
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
news_df = pd.read_csv('train.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + ' ' + news_df['title']
X = news_df.drop('label', axis=1)
y = news_df['label']

# Define stemming function
ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Apply stemming function to content column
news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values
vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Fit logistic regression model
model = LogisticRegression()
model.fit(X_train,Y_train)


# website
# Website with Custom Styles
st.markdown(
    """
    <style>
    body {
        margin: 0;
        padding: 0;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-image: url('https://source.unsplash.com/1600x900/?news,media');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color:#000;
        padding: 20px;
    }
    .overlay {
        background: rgba(0, 0, 0, 0.6);
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 1;
    }
    .main-container {
        position: relative;
        z-index: 2;
        padding: 30px;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .main-title {
        font-size: 3rem;
        text-align: center;
        color:black;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
    }
    .input-box {
        margin: 20px auto;
        width: 70%;
        font-size: 1.2rem;
        padding: 10px;
    }
    .prediction-result {
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        color: #10b981;
        margin-top: 20px;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.6);
    }
    .prediction-fake {
        color: #ef4444;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="overlay"></div>', unsafe_allow_html=True)
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Fake News Detector</h1>', unsafe_allow_html=True)

input_text = st.text_input('Enter news Article:', placeholder='Type or paste your news article here...', key="input-box")

def prediction(input_text):
    input_data = vector.transform([input_text])
    prediction = model.predict(input_data)
    return prediction[0]

if input_text:
    pred = prediction(input_text)
    if pred == 1:
        st.markdown('<div class="prediction-result prediction-fake">The News is Fake</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="prediction-result">The News is Real</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)