import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
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

def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()

dataset = nlp.load_dataset("emotion")

train = dataset['train']
val = dataset['validation']
test = dataset['test']

def get_tweets(data):
    tweets = [x["text"] for x in data]
    labels = [x["label"] for x in data]
    return tweets, labels

train_tweets, train_labels = get_tweets(train)

# tokenizing data

from tensorflow.keras.preprocessing.text import Tokenizer

# less common words get less common word token

tokenizer = Tokenizer(num_words=10000, oov_token="<UNKNOWN>")
tokenizer.fit_on_texts(train_tweets)

tokenizer.texts_to_sequences(train_tweets[0])

# padding and truncating sequences 

lengths = [len(t.split(' ')) for t in train_tweets]
plt.hist(lengths, bins = len(set(lengths)))
plt.show()

# histogram shows that most tweets are between 10 and 20 words long

maxlength = 50 # if above 50 words, truncate, if below 50 words, pad with zeros

from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_sequences(tokenizer, tweets):
    sequences = tokenizer.texts_to_sequences(tweets)
    padded = pad_sequences(sequences, maxlen=maxlength, padding='post', truncating='post')
    return padded

padded_train = get_sequences(tokenizer, train_tweets)

# preparing labels

classes = set(train_labels)
print("Classes:", classes) # 6 classes

plt.hist(train_labels, bins = 11)
plt.show()

class_to_index = {c: i for i, c in enumerate(classes)}
index_to_class = {v: k for k, v in class_to_index.items()}

names_to_ids = lambda labels: np.array([class_to_index.get(x) for x in labels]) 

train_labels = names_to_ids(train_labels)

# creating the model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=maxlength),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(64, activation='softmax'),
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

val_tweets, val_labels = get_tweets(val)
val_sequences = get_sequences(tokenizer, val_tweets)
val_labels = names_to_ids(val_labels)

h = model.fit(
    padded_train, 
    train_labels, 
    validation_data=(val_sequences, val_labels), 
    epochs=20, 
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
)

show_history(h)

# evaluating the model

test_tweets, test_labels = get_tweets(test)
test_sequences = get_sequences(tokenizer, test_tweets)
test_labels = names_to_ids(test_labels)

_ = model.evaluate(test_sequences, test_labels)

print("Sentence:", test_tweets[0])
print("True label:", index_to_class.get(test_labels[0]))

predictions = model.predict(np.expand_dims(test_sequences[0], axis=0))
predicted_label = index_to_class[np.argmax(predictions).astype("uint8")]



