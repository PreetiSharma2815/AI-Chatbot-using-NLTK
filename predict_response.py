#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
ignore_words = ['?', '!',',','.', "'s", "'m"]


import json
import pickle

import numpy as np
import random

# Model Train Lib
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

from train_bot import get_stem_words

model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Load data files
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))

print("Hi I am Stella, How Can I help you?")
user_input = input("Type your message here:")


def preprocess_user_input(user_input):
    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
  
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []

    # Input data encoding 
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
  
    return np.array(bag)
    
def bot_class_prediction():
    inp = preprocess_user_input(user_input)
    prediction = model.predict(inp)

    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response():

   predicted_class_label =  bot_class_prediction()

   predicted_class = classes[predicted_class_label]

   for intent in intents['intents']:
      
    if intent['tag']==predicted_class:
       
        bot_response = random.choice(intent['responses'])
    
        return bot_response
    
bot_response = bot_response()

