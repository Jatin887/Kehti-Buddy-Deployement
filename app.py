# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup, jsonify
import numpy as np
from nltk.tokenize import word_tokenize
import pandas as pd
import requests
import config
import pickle
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from flask_cors import CORS
import os
import re
nltk.download('punkt')
nltk.download('stopwords')
from keras.models import load_model

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

def clean_sentence(sentence: str) -> list:
    # Remove the review tag
    tags = re.compile("(<review_text>|<\/review_text>)")
    sentence = re.sub(tags, '', sentence)
    # lower case
    sentence = sentence.lower()
    # Remove emails and urls
    email_urls = re.compile("(\bhttp.+? | \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)")
    sentence = re.sub(email_urls, '', sentence)
    # Some used '@' to hide offensive words (bla -> bl@)
    ats = re.compile('@')
    sentence = re.sub(ats, 'a', sentence)
    # Remove Punctuation 
    # punc = re.compile("[!\"\#\$\%\&\'\(\)\*\+,\-\.\/\:;<=>\?\[\\\]\^_`\{\|\}\~]")
    punc = re.compile("[^\w\s(\w+\-\w+)]")
    sentence = re.sub(punc, '', sentence)
    # Remove stopwords and tokenize
    # sentence = sentence.split(sep=' ')
    sentence = word_tokenize(sentence)
    sentence = [word for word in sentence if not word in stopwords.words()]
    # Stemming (Returning to root)
    # stemmer = PorterStemmer()
    # sentence = [stemmer.stem(word) for word in sentence]
    return sentence
# or 175 if we took all review sizes into consideration
x_train = open('x_train_1.txt','r')
x_train = eval(x_train.read())
vocab = set()
for sentence in x_train:
    for word in sentence:
        vocab.add(word)

vocab.add('')
print("Vocab size:", len(vocab))

word2id = {word:id for  id, word in enumerate(vocab)}
def encode_sentence(old_sentence):
    encoded_sentence = []
    dummy = word2id['']
    for word in old_sentence:
        try:
            encoded_sentence.append(word2id[word])
        except KeyError:
            encoded_sentence.append(dummy) # the none char
    return encoded_sentence
MAX_SEQ_LEN = 125
dummy = word2id['']
def lstm_predict(sentence:str):
    sentence = clean_sentence(sentence)
    # Encode sentence
    # lstm_model_path = 'models/lstm.pkl'
    # lstm_model = pickle.load(
    # open(lstm_model_path, 'rb'))

    lstm_model = load_model('models/lstm_model.h5')
    ready_sentence = encode_sentence(sentence)
    # Padding sentence
    ready_sentence = pad_sequences(sequences = [ready_sentence], 
                                     maxlen=MAX_SEQ_LEN,
                                     dtype='int32', 
                                     padding='post',
                                     truncating='post',
                                     value = dummy)
    # Predict
    prediction = round(lstm_model.predict(ready_sentence)[0][0])
    if prediction==0:
        return "Negative Review"
    elif prediction==1:
        return "Positive Review"
    else:
        print('Error')

# Loading crop recommendation model




# =========================================================================================

# Custom functions for calculations




# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)
CORS(app)

port = os.environ.get('PORT', 80)

@ app.route('/')
def home():
    title = 'Sentiment Analysis'
    return render_template('predection.html', title=title)

@ app.route('/sentiment-predict', methods=['POST'])
def sentiment_prediction():
    title = 'Sentiment Analysis'

    if request.method == 'POST':
        data = request.form['sentence']
        my_prediction = lstm_predict(data)
        return render_template('sentiment-result.html', prediction=my_prediction, title=title)
         # return jsonify({"final_prediction": final_prediction})

    else:
        return render_template('try_again.html', title=title)


if __name__ == '__main__':
     app.run(debug=False)
