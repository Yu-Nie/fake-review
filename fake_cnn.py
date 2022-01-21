#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import os
import pickle
import sys

import pandas as pd
from keras import layers, metrics, callbacks
from keras.backend import clear_session
from keras.models import Sequential, model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_args():
    parser = argparse.ArgumentParser(description='get parameters')
    parser.add_argument('-e', '--test', help='Test attributes (to predict)')
    parser.add_argument('-n', '--train', help='Train data')
    parser.add_argument('-v', '--validation', help='Validation data')

    args = parser.parse_args()
    return args


def run(arguments):
    opt = get_args()
    train_path = opt.train
    validation_path = opt.validation
    test_path = opt.test

    Train = True if train_path is not None else False
    Test = True if test_path is not None else False
    Validation = True if validation_path else False

    if Train and Validation:
        # train_path = "reviews_train.csv"
        file_train = pd.read_csv(train_path, quotechar='"', usecols=[0, 1, 2, 3],
                                 dtype={'real review?': int, 'category': str, 'rating': int, 'text_': str})
        sentence_train = file_train['text_']
        y_train = file_train['real review?']

        # validation_path = "reviews_validation.csv"
        file_validate = pd.read_csv(validation_path, quotechar='"', usecols=[0, 1, 2, 3],
                                    dtype={'real review?': int, 'category': str, 'rating': int, 'text_': str})
        sentence_validate = file_validate['text_']
        y_validate = file_validate['real review?']

        # return most 5000 frequent words when call texts_to_sequences
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(sentence_train)
        Xcnn_train = tokenizer.texts_to_sequences(sentence_train)
        with open('tokenizer.pk', 'wb') as fout:
            pickle.dump(tokenizer, fout)
        Xcnn_validation = tokenizer.texts_to_sequences(sentence_validate)

        vocab_size = len(tokenizer.word_index) + 1
        maxlen = 100
        Xcnn_train = pad_sequences(Xcnn_train, padding='post', maxlen=maxlen)
        Xcnn_validation = pad_sequences(Xcnn_validation, padding='post', maxlen=maxlen)
        embedding_dim = 200
        model = Sequential()
        # add word embedding
        model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
        model.add(layers.Conv1D(128, 5, activation='relu'))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(10, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=[metrics.AUC()])
        # model.summary()
        earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                                mode="min", patience=5,
                                                restore_best_weights=True)
        clear_session()
        model.fit(Xcnn_train, y_train,
                  epochs=25,
                  verbose=2,
                  validation_data=(Xcnn_validation, y_validate),
                  batch_size=100,
                  callbacks=[earlystopping])
        # serialize model to JSON
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")

        loss, accuracy = model.evaluate(Xcnn_train, y_train, verbose=0)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(Xcnn_validation, y_validate, verbose=0)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    elif Test:
        tokenizer = pickle.load(open('tokenizer.pk', 'rb'))
        # test_path = "reviews_test_attributes.csv"
        file_test = pd.read_csv(test_path, quotechar='"', usecols=[0, 1, 2, 3],
                                dtype={'real review?': int, 'category': str, 'rating': int, 'text_': str})
        sentence_test = file_test['text_']
        x_test = tokenizer.texts_to_sequences(sentence_test)
        Xcnn_test = pad_sequences(x_test, padding='post', maxlen=100)

        # load json and create model
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")

        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[metrics.AUC()])
        y_hat = loaded_model.predict(Xcnn_test)

        # save predictions result
        header = ['ID', 'real review?']
        with open('prediction.csv', 'w', newline='') as fw:
            writer = csv.writer(fw)
            writer.writerow(header)
            for i, y in enumerate(y_hat):
                writer.writerow([i, y[0]])


if __name__ == "__main__":
    run(sys.argv[1:])
