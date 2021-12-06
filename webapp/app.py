from flask import Flask,request, render_template, jsonify
from flask import url_for, make_response, request
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from collections import defaultdict
tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
import nltk
import json
import pickle
import numpy as np
import re
import string
import pandas as pd
import joblib
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__, template_folder='templates')
with open('/Users/aryajeshkumar/Desktop/webapp/model/vectorizer.pkl', 'rb') as f:
	vectorizer = joblib.load(f)

with open('/Users/aryajeshkumar/Desktop/webapp/model/sentiment_classifier.pkl', 'rb') as f:
	sentiment_classifier = joblib.load(f)

def preprocess_text(text):
    # Removing html tags
    sentence = remove_tags(text)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence.lower())
    
    # stopwords removal
    sentence = stop_words(sentence)
    
    # lemmatization 
    sentence = lemmatize_words(sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def stop_words(text):
    filtered_sentence = []
    tokens = nltk.word_tokenize(text)
    for w in tokens:
        if w not in stopwords.words('english') and w.isalpha():
            filtered_sentence.append(w)
            str1=' '.join(str(e) for e in filtered_sentence)
    return str1

def lemmatize_words(text):
    tokens = nltk.word_tokenize(text)
    lmtzr = WordNetLemmatizer()
    tagged=nltk.pos_tag(tokens)
    filtered_sentence = []
    for token, tag in tagged:
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        filtered_sentence.append(lemma)
        str2=' '.join(str(e) for e in filtered_sentence)
    return str2 

	
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
		if request.method == 'POST':
			data = request.form.get('message')
			text = preprocess_text(data)
			vect = vectorizer.transform([text])
			my_prediction = sentiment_classifier.predict(vect)
			my_confidence = sentiment_classifier.predict_proba(vect)
			if my_prediction[0] == 1:
				tag="Positive"
				confi=my_confidence[0][1]
			else: 
				tag="Negative"
				confi=my_confidence[0][0]
			return render_template('index.html',preprocessed = text,prediction = tag,confidence=confi)

if __name__ == '__main__':
	app.run(debug=True, port=8080)
	