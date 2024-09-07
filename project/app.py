# Import necessary libraries
from flask import Flask, render_template, request
import numpy as np
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model_clf = joblib.load('trained_model1.pkl')

# Load pre-trained GloVe embeddings
# Assuming you have already downloaded the GloVe embeddings file
# and saved it as glove.6B.100d.txt in the same directory as your app.py
def load_glove_embeddings(embeddings_file):
    embeddings_index = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings_file = "glove.6B.100d.txt"
glove_embeddings = load_glove_embeddings(glove_embeddings_file)

# Define stopwords
stop_words = set(stopwords.words('english'))

# Define your clean_tweet function
def clean_tweet(tweet):
    tweet = re.sub("#", "", tweet)  # Removing '#' from hashtags
    tweet = re.sub("[^a-zA-Z#]", " ", tweet)  # Removing punctuation and special characters
    tweet = re.sub(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', "<URL>", tweet)
    tweet = re.sub('http', '', tweet)
    tweet = re.sub(" +", " ", tweet)
    tweet = tweet.lower()
    tweet = word_tokenize(tweet)
    return_tweet = []
    for word in tweet:
        if word not in stop_words:
            return_tweet.append(word)
    return return_tweet

# Function to extract features from tweet using pre-trained GloVe embeddings
def get_features(tweet):
    features = []
    for word in tweet:
        if word in glove_embeddings:
            features.append(glove_embeddings[word])
    return np.mean(features, axis=0) if features else np.zeros(100)  # Assuming 100-dimensional GloVe embeddings

# Function to predict class label for input tweet/comment
def predict_class(input_tweet, class_labels):
    preprocessed_input_tweet = clean_tweet(input_tweet)
    input_features = get_features(preprocessed_input_tweet)
    class_probabilities = model_clf.predict_proba(input_features.reshape(1, -1))

    predicted_class_index = np.argmax(class_probabilities)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

# Define a route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    class_labels = ["0", "1", "2"]  # Define your class labels here
    predicted_class = predict_class(comment, class_labels)
    if predicted_class == "0":
        prediction = "Hate Speech"
    elif predicted_class == "1":
        prediction = "Offensive Language"
    else:
        prediction = "Normal Language"
    return render_template('result.html', prediction=prediction)

# Define a route to render the form
@app.route('/')
def form():
    return render_template('index.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
