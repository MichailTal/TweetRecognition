# Necessary Imports

import matplotlib.pyplot as plt
import nlp
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Plotting the Epoch History

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim(0., 1.)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Plottting the Confusion Matrix


def show_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()


# TF Version for comparison (not necessary)
print('Using Tensorflow Version: ', tf.__version__)

# Importing and accessing the Data

dataset = nlp.load_dataset('emotion')

train = dataset['train']
val = dataset['validation']
test = dataset['test']


def get_tweet(data):
    tweets = [x['text'] for x in data]
    labels = [x['label'] for x in data]

    return tweets, labels


tweets, labels = get_tweet(train)

# print(tweets[1], labels[1])   --> Just testing the access

# Tokenizing the tweets

tokenizer = Tokenizer(num_words=10000,
                      oov_token='<UNK>')  # defining the out-of-vocabulary-token with the 10.000 most used words
tokenizer.fit_on_texts(tweets)

# (tokenizer.texts_to_sequences([tweets[0]])) --> test case

# Padding and Truncating

lengths = [len(t.split(' ')) for t in tweets]  # length of tweets separated by ' '
maxlen = 55  # truncating tweets longer than 55 words


def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, truncating='post', padding='post', maxlen=maxlen)
    return padded


padded_train_seq = get_sequences(tokenizer, tweets)

# print(padded_train_seq[0]) --> test case

# Labelling

classes = set(labels)

class_to_index = dict((c, i) for i, c in enumerate(classes))
index_to_class = dict((v, k) for k, v in class_to_index.items())
names_to_ids = lambda labels: np.array([class_to_index.get(x) for x in labels])

train_labels = names_to_ids(labels)
# print(train_labels[25])

# Model

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.compile(

    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# model.summary()


# Training the Model

val_tweets, val_labels = get_tweet(val)
val_seq = get_sequences(tokenizer, val_tweets)
val_labels = names_to_ids(val_labels)

h = model.fit(
    padded_train_seq, train_labels,
    validation_data=(val_seq, val_labels),
    epochs=20,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
    ]
)

# Evaluation

show_history(h)

test_tweets, test_labels = get_tweet(test)
test_seq = get_sequences(tokenizer, test_tweets)
test_labels = names_to_ids(test_labels)

_ = model.evaluate(test_seq, test_labels)

i = random.randint(0, len(test_labels) - 1)

print('Sentence: ', test_tweets[i])
print('Emotion: ', index_to_class[i])

p = model.predict(np.expand_dims(test_seq[i], axis=0))[0]
pred_class = index_to_class[np.argmax(p).astype('uint8')]

print('Predicted Emotion: ', pred_class)

# pred = model.predict(test_seq)
# show_confusion_matrix(test_labels, pred, list(classes))
