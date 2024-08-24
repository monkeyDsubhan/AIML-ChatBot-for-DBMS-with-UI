from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize
import random
import json
import os

app = Flask(__name__, template_folder=r'C:\Users\HP\Desktop\Chatbot\templates')

# Load the chatbot model and data
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json', encoding="utf8").read())

# Define the ignore_words variable
ignore_words = ['?', '!', '.', ',']

# Initialize NLTK
nltk.download('punkt')

# Define the chatbot functions
def predict_class(message, model):
    try:
        # Tokenize the message
        tokens = nltk.word_tokenize(message)
        tokens = [word for word in tokens if word not in ignore_words]

        # Create a bag of words
        bag = [0] * len(words)
        for word in tokens:
            for i, w in enumerate(words):
                if w == word:
                    bag[i] = 1

        # Reshape the bag of words to match the model's input shape
        bag = np.array(bag).reshape((1, len(words)))

        # Make a prediction using the model
        res = model.predict(bag)
        res_index = np.argmax(res)
        print(f"Predicted class index: {res_index}")
        return res_index
    except Exception as e:
        print(f"Error occurred in predict_class: {e}")
        return None

def get_response(ints, intents):
    try:
        tag = classes[ints]
        print(f"Tag: {tag}")
        for intent in intents['intents']:
            if intent['tag'] == tag:
                response = random.choice(intent['responses'])
                print(f"Response: {response}")
                return response
    except Exception as e:
        print(f"Error occurred in get_response: {e}")
        return None

def chatbot(message):
    try:
        ints = predict_class(message, model)
        print(f"Predicted class index in chatbot: {ints}")
        res = get_response(ints, intents)
        print(f"Response in chatbot: {res}")
        return res
    except Exception as e:
        print(f"Error occurred in chatbot: {e}")
        return None

# Define the Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    message = request.form.get('message')
    if message is None or message.strip() == '':
        return jsonify({'error': 'Invalid message'}), 400
    response = chatbot(message)
    if response is None:
        return jsonify({'error': 'Error occurred'}), 500
    return jsonify({'response': response})

# Start the Flask development server
if __name__ == '__main__':
    app.run(debug=True)