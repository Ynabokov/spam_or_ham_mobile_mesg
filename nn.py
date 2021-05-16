import re
import nltk
import itertools
import string
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pan
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras_metrics
from keras.models import Model, Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K
from nltk.stem import SnowballStemmer

EPOCHS_NO = 10
FILE_PATH = 'data/data.csv'

stemmer = SnowballStemmer("english")

wordnet_lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))

def remove_appostophes(sentence):
    APPOSTOPHES = {"s" : "is", "re" : "are", "t": "not", "ll":"will","d":"had","ve":"have","m": "am"}
    words = nltk.tokenize.word_tokenize(sentence)
    final_words=[]
    for word in words:
        broken_words=word.split("'")
        for single_words in broken_words:
            final_words.append(single_words)
    reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in final_words]
    reformed = " ".join(reformed)
    return reformed
def remove_punctuations(my_str):
    punctuations = '''!()-[]{};:'"\,./?@#$%^&@*_~'''
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct
def clean_data(sentence):
    # ## removing web links
    s = [ re.sub(r'http\S+', '', sentence.lower())]
    ## removing words like gooood and poooor to good and poor
    s = [''.join(''.join(s)[:2] for _, s in itertools.groupby(s[0]))]
    ## removing appostophes
    s = [remove_appostophes(s[0])]
    ## removing punctuations from the code 
    s = [remove_punctuations(s[0])]
    return s[0]



training_dataset = pan.read_csv(FILE_PATH, delimiter=',' , encoding="latin-1")
training_dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
training_dataset.columns=["labels","message"]

Labels = training_dataset["labels"]
training_dataset.drop(['labels'], axis = 1, inplace = True)

for index in range(0,len(training_dataset["message"])):
    training_dataset.loc[index,"message"] = clean_data(training_dataset["message"].iloc[index])

Messages = training_dataset["message"]
le = LabelEncoder()
Labels = le.fit_transform(Labels)
Labels = Labels.reshape(-1,1)

X_train,X_test,Y_train,Y_test = train_test_split(Messages,Labels,test_size=0.5)


max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.summary()

model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])


history_ltsm = model.fit(sequences_matrix,Y_train, batch_size=128, epochs=10,
          validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

model.save('RNN_Spam.h5') 

def getPrediction (prediction):
    if prediction >= 1:
    	print ("Ham")
    else:
    	print("Spam")

accuracy = history_ltsm.history['accuracy']
val_accuracy = history_ltsm.history['val_accuracy']
loss = history_ltsm.history['loss']
val_loss = history_ltsm.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, '-', color='gold', label='training accuracy')
plt.plot(epochs, val_accuracy, '-', color='darkcyan', label='validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, '-', color='crimson', label='training loss')
plt.plot(epochs, val_loss,  '-', color='navy', label='validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

test_sequences = tok.texts_to_sequences(X_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)

test_sequences1 = tok.texts_to_sequences("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate).")
test_sequences_matrix1 = sequence.pad_sequences(test_sequences1,maxlen=max_len)

test_sequences2 = tok.texts_to_sequences("Hi, how are u?")
test_sequences_matrix2 = sequence.pad_sequences(test_sequences2,maxlen=max_len)

results = model.evaluate(test_sequences_matrix,Y_test)
prediction = model.predict(test_sequences_matrix1)[0]
prediction2 = model.predict(test_sequences_matrix2)

getPrediction(np.argmax(prediction))
getPrediction(np.argmax(prediction2))


precision = results[2]
recall = results[3]
f_score = 2*((precision*recall)/(precision+recall))
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(results[0],results[1]))
print('Test set\n  Precision: {:0.3f}\n  Recall: {:0.3f}\n  F-Score: {:0.3f}'.format(precision, recall, f_score))
