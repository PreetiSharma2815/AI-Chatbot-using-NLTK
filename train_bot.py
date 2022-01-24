#Text Data Preprocessing Lib
import nltk
nltk.download('punkt')
nltk.download('wordnet')


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle

import numpy as np
import random

#Model Train Lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


words=[] #list of unique roots words in the data
classes = [] #list of unique tags in the data
pattern_word_tags_list = [] #list of the pair of (['words', 'of', 'the', 'sentence'], 'tags')

ignore_words = ['?', '!',',','.', "'s", "'m"]


train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words

# stem_words = ['advers', 'anyon', 'awesom', 'behavior', 
# 'blood', 'bye', 'caus', 'chat', 'check', 'could', 'data', 'day', 'detail',
#  'dont', 'drug', 'entri', 'find', 'give', 'good', 'goodby', 'hello', 'help',
#   'hey', 'hi', 'histori', 'hola', 'hospit', 'how', 'i', 'id', 'is', 'later',
#    'list', 'load', 'locat', 'log', 'look', 'lookup', 'manag', 'modul', 'nearbi',
#     'next', 'nice', 'offer', 'open', 'patient', 'pharmaci', 'pressur', 'provid', 
#     'reaction', 'relat', 'result', 'search', 'see', 'show', 'suitabl', 'support',
#      'task', 'thank', 'that', 'till', 'time', 'transfer', 'want', 'what', 'which']
 
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in intents['intents']:

        # Add all patterns and tags to a list
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                        
            pattern_word_tags_list.append((pattern_word, intent['tag']))
              
    
        # Add all tags to the classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
    stem_words = get_stem_words(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words, classes, pattern_word_tags_list


# Training Dataset: 
# Input Text----> as Bag of Words 
# Tags-----------> as Label

def bag_of_words_encoding(stem_words, pattern_word_tags_list):
    bag = []

    for word_tags in pattern_word_tags_list:
        # example: word_tags = (['hi', 'there'], 'greetings']

        pattern_words = word_tags[0] # ['hi', 'there']
        bag_of_words = []

        # Input data encoding 
        for word in stem_words:            
            if word in pattern_words:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
    
    bag.append(bag_of_words)

    return np.array(bag)

def class_label_encoding(classes, pattern_word_tags_list):
    
    labels = []

    for word_tags in pattern_word_tags_list:

        # Start with list of 0s
        labels_encoding = list([0]*len(classes))  

        # example: word_tags = (['hi', 'there'], 'greetings']

        tag = word_tags[1]   # 'greetings'

        tag_index = classes.index(tag)

        # Labels Encoding
        labels_encoding[tag_index] = 1

    labels.append(labels_encoding)

    return np.array(labels)

def preprocess_train_data(stem_words, classes, pattern_word_tags_list):
   
    train_x = bag_of_words_encoding(stem_words, pattern_word_tags_list)
    train_y = class_label_encoding(classes, pattern_word_tags_list)
    
    return train_x, train_y


def train_bot_model(train_x, train_y):
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    # Compile Model
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Fit & Save Model
    history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=True)
    model.save('chatbot_model.h5', history)

    print("Model File Created & Saved")


# Calling methods

stem_words, classes, pattern_word_tags_list = create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words)

pickle.dump(stem_words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

train_x, train_y = preprocess_train_data(stem_words, classes, pattern_word_tags_list)

train_bot_model(train_x, train_y)

