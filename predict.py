#this is a model without the csv file.


import re
import numpy as np
import joblib
from googletrans import Translator
from transformers import pipeline
from gensim.models import Word2Vec

# Load models
translator = Translator()
sentiment_model = pipeline("sentiment-analysis")
lgb_model = joblib.load("predictor/models/mental_health_classifier_lgb.pkl")  # Adjust path if necessary
w2v_model = joblib.load("predictor/models/word2vec_model.pkl")

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s\?\!]", "", text)
    text = text.lower().strip()
    return text

# Detect and translate using Google Translate
def detect_and_translate(text):
    try:
        detected_lang = translator.detect(text).lang
        if detected_lang != "en":
            text = translator.translate(text, dest="en").text
        return detected_lang, text
    except Exception as e:
        return "unknown", text

# Perform sentiment analysis
def analyze_sentiment(text):
    try:
        result = sentiment_model(text)
        return result[0]["label"], result[0]["score"]
    except Exception:
        return "UNKNOWN", 0.0

# Convert text to vector using Word2Vec
def vectorize_text(text):
    words = text.split()
    vector = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
    return sum(vector) / len(vector) if vector else np.zeros(100)

# Predict using models
def predict_mental_health(text):
    cleaned_text = clean_text(text)
    detected_lang, translated_text = detect_and_translate(cleaned_text)

    sentiment, score = analyze_sentiment(translated_text)

    if sentiment == "POSITIVE" and score > 0.85:
        return "✅ No issues detected. Stay positive and take care of yourself! 💙"

    vectorized_text = np.array([vectorize_text(translated_text)])
    prediction = lgb_model.predict(vectorized_text)
    return f"⚠️ Early Sign of Detected Mental Health Issue: {prediction[0]}. Stay positive and take care of yourself!"





# # with csv file.
# import pandas as pd
# import numpy as np
# import re
# import joblib
# from gensim.models import Word2Vec
# from googletrans import Translator
# from transformers import pipeline

# # Load models and data
# df = pd.read_csv("predictor/data/filtered_cleaned_data_final2.csv")
# w2v_model = joblib.load("predictor/models/word2vec_model.pkl")
# classifier_model = joblib.load("predictor/models/mental_health_classifier_lgb.pkl")

# # Initialize translator and sentiment model
# translator = Translator()
# sentiment_model = pipeline("sentiment-analysis")

# # Tokenize if model not available
# if not w2v_model:
#     tokenized_text = [tweet.split() for tweet in df["cleaned_text"]]
#     w2v_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=2, workers=4)

# # Text cleaning function
# def clean_text(text):
#     text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
#     text = re.sub(r"[^a-zA-Z0-9\s\?\!]", "", text)
#     text = text.lower().strip()
#     return text

# # Detect and translate language
# def detect_and_translate(text):
#     try:
#         detected_lang = translator.detect(text).lang
#         if detected_lang != "en":
#             translated = translator.translate(text, dest="en").text
#             return detected_lang, translated
#     except Exception:
#         return "unknown", text
#     return detected_lang, text

# # Perform sentiment analysis
# def analyze_sentiment(text):
#     result = sentiment_model(text)
#     return result[0]["label"], result[0]["score"]

# # Convert text to vector using Word2Vec
# def vectorize_text(text, model):
#     words = text.split()
#     vector = [model.wv[word] for word in words if word in model.wv]
#     return sum(vector) / len(vector) if vector else np.zeros(100)

# # Predict mental health issue
# def predict_mental_health(text):
#     cleaned_text = clean_text(text)
#     detected_lang, translated_text = detect_and_translate(cleaned_text)
#     sentiment, score = analyze_sentiment(translated_text)

#     if sentiment == "POSITIVE" and score > 0.85:
#         return "✅ No issues detected. Stay positive and take care of yourself!"
    
#     vectorized_text = vectorize_text(translated_text, w2v_model)
#     vectorized_text = np.array([vectorized_text])
#     predicted_label = classifier_model.predict(vectorized_text)
#     return f"⚠️ Early Sign of Detected Mental Health Issue: {predicted_label[0]}. Stay positive and take care of yourself!"
