# Step 1: Import Necessary Libraries

# import necessary libraries
import warnings
warnings.filterwarning("ignore")
import nltk
from nltk.stem import WordNetLemmatiizer
import json
import pickle

import numpy as np
from tensorflow.keras.model import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

from keras.models import load_model

# Step 2: Data Pre-Processing

# preprocessing the json data
# tokenization
import random

for intent in intents['intents']:
    for pattern in intents['patterns']:

        # tokenize each word
        w = nltk.work_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))

        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# testing tokenization
print("~This is words list~")
print(words[3:5])
print("-" * 50)
print("~This is documents list~")
print(documents[3:5])
print("-" * 50)
print("~This is classes list~")
print(classes[3:10])

# lemmatize, lower each word and remove duplicates

lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower()) for w in words if not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# documents = combination between parents and intents

print("~Document Length")
print(len(documents), "documents\n\n")
print("-" * 100)

# classes = intents
print("~Class Length")
print(len(classes), "classes\n\n", classes)
print("-" * 100)

# words = all words, vocabulary
print("~Words Length~")
print(len(words), "unique lemmatized words\n\n", words)

# creating a pickle file to store the Python objects which we will use while predicting
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))


# Step 3: Creating Training Data

# initialize training data
training = []

# create empty array for output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:

    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]

    # lemmatize each words - create base words, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    # create our bag of words array with 1, if words match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle features and convert it into multiple arrays
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print("Training data created")


# Step 4. Creating a Neural Network Model

# create NN model to predict the responses
model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]), ), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Droupout(0.5))
model.add(Dense(len(train_y[0]), activation = 'softmax'))

# compile model. stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot.h5', hist) # we will pickle this model to use in the future

print("\n")
print("-" * 50)
print("\nModel created successfully.")


# Step 5. Create functions to take user input, pre-process the input, predict the class, and get the response

# define function to clean the user input
def clean_up_sentence(sentence):

    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)

    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence words]
    return sentence_words

# define function to return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details = True)

    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)

    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:

                # assign 1 if current word is in the vocaubulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))

# define function to predict the target class
def predict_class(sentence, model):

    # filter out predictions below a threshold
    p = bow(sentence, words, show_details = False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error]

    # sort by strength of probability
    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# define function to get the response from the model
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

# define function to predict the class and get the response
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res

# define function to start the chatbot which will continue until the user types "end"
def start_chat():
    print("Bot: This is Sophie, your personal assistant.\n\n")
    while True:
        inp = str(input()).lower()
        if inp.lower() == "end":
            break
        if inp.lower() == '' or inp.lower() == '*':
            print("Please re-phrase your query.")
            print("-" * 50)
   