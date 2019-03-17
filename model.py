# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:28:11 2019

@author: Baptiste
"""

from __future__ import print_function, division
import tensorflow as tf

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D, SimpleRNN, GRU, LSTM
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

from functions import * # the file containing the functions is secret



flags = tf.app.flags

flags.DEFINE_string('dir_path', 'data/data_brutes/', 'The dir of the datas location')
flags.DEFINE_integer('batch_size',50, "Batch size")
flags.DEFINE_integer('nb_epochs',100, "Nb of epochs")
flags.DEFINE_boolean('save_weights',False, "save the model's weigths or not")
flags.DEFINE_boolean('from_dir',True, "If True then load the datas from data_brutes dir")
flags.DEFINE_integer('procede',1, "The dictionnary of classes to predict to use in the model")
flags.DEFINE_string('model','rnn', "the type of model to use (mlp, rnn, lstm, gru)")
flags.DEFINE_boolean('save_fig',False, "save the figure plotted or not")

FLAGS = flags.FLAGS


"""
model_MLP:
	function that create a MLP model
@params:
	max_words, the number maxof word from the dict
	max_len, the number max of worf per sentence
	num_classes, the number of classes to predict
"""
def model_MLP(max_words,maxlen,num_classes):
    model = Sequential()
    model.add(Embedding(max_words, 32, input_length=maxlen))
    # model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    
    return model

"""
model_RNN:
	function that create a RNN model
@params:
	max_words, the number maxof word from the dict
	max_len, the number max of worf per sentence
	num_classes, the number of classes to predict
"""
def model_RNN(max_words,maxlen,num_classes):
    model = Sequential()
    model.add(Embedding(max_words, 32, input_length=maxlen))
    # model.add(Dropout(0.5))
    model.add(SimpleRNN(64, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=RMSprop(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    
    return model

"""
model_LSTM:
	function that create a LSTM model
@params:
	max_words, the number maxof word from the dict
	max_len, the number max of worf per sentence
	num_classes, the number of classes to predict
"""
def model_LSTM(max_words,maxlen,num_classes):
    model = Sequential()
    model.add(Embedding(max_words, 32, input_length=maxlen))
    # model.add(Dropout(0.5))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=RMSprop(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model

"""
model_GRU:
	function that create a GRU model
@params:
	max_words, the number maxof word from the dict
	max_len, the number max of worf per sentence
	num_classes, the number of classes to predict
"""
def model_GRU(max_words,maxlen,num_classes):
    model = Sequential()
    model.add(Embedding(max_words, 32, input_length=maxlen))
    # model.add(Dropout(0.5))
    model.add(GRU(64, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model

def __main__():
    
    os.chdir(os.path.dirname(os.path.realpath(__file__))+'/')
    os.chdir("../../")
    
    cur_path = os.getcwd()+"/"
    
    try:
        DATA_IN = os.environ["DATA_IN_TRAINING"]
    except:
        print("environement variable DATA_IN_TRAINING failed to be loaded ...")
        DATA_IN = "..."# path to the data folder
    
    
    if FLAGS.from_dir: 
        description, procede = load_train_dir(cur_path + FLAGS.dir_path)
    else:
        try:
            description, procede = load_train_dir(DATA_IN)
        except:
            print("wrong repertory for DATA_IN_TRAINING")
    
    word_dictionnary = dict_word(description) # creating the dictionnary of the words used in the model
    
    if FLAGS.procede == 1:
        proc_codage = procede_codage
        num_classes = len(np.unique([int(ii) for ii in procede_codage.values()]))
    elif FLAGS.procede == 2:
        proc_codage = procede_codage_2
        num_classes = len(np.unique([int(ii) for ii in procede_codage_2.values()]))
    elif FLAGS.procede == 3:
        proc_codage = procede_codage_3
        num_classes = len(np.unique([int(ii) for ii in procede_codage_3.values()]))
    
    max_length = 16 # arbitrary, we never had more than 16 words in a sentence
    
    X = transform_X(description,word_dictionnary,max_length) # transforming the text with some magic
    Y = transform_Y(procede, proc_codage) # transforming the target with some magic
    
    maxlen = max_length # hyperparameter of the model
    max_words = 2000 # hyperparameter of the model
    
    labelsnb = range(0,len(proc_codage.keys())+1)
    class_weight = {}
    for index, label in enumerate(labelsnb):
        class_weight[index] = len([x for x in procede if proc_codage[x]==label])/len(Y)
    
    if FLAGS.model == "mlp":
        print("using MLP")
        model = model_MLP(max_words,maxlen,num_classes)
    elif FLAGS.model == "rnn":
        print("using RNN")
        model = model_RNN(max_words,maxlen,num_classes)
    elif FLAGS.model == "lstm":
        print("using LSTM")
        model = model_LSTM(max_words,maxlen,num_classes)
    elif FLAGS.model == "gru":
        print("using GRU")
        model = model_GRU(max_words,maxlen,num_classes)
    
    
    callbacks = [
        #ReduceLROnPlateau(),
        #ModelCheckpoint(filepath='model-simple.h5', save_best_only=True)
    ]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    
    history = model.fit(X_train, Y_train,
                        class_weight=class_weight,
                        epochs=FLAGS.nb_epochs,
                        batch_size=FLAGS.batch_size,
                        validation_split=0.2,
                        callbacks=callbacks,
                        initial_epoch=0)
    
    score = model.evaluate(X_test, Y_test,batch_size=32)
    print('Test accuracy:', score[1])
    
	# figname takes name like 'model_rnn_200epochs_38_classes.png'
    figname = cur_path + 'src/sorties_graphiques/model_' + FLAGS.model + '_' + str(FLAGS.nb_epochs) + 'epochs_' + str(FLAGS.procede) + 'proc.png'
    
    maxtrain = max(history.history['categorical_accuracy']) # the max accurary get by training
    maxtest = max(history.history['val_categorical_accuracy']) # the max accuracy get by testing
    plt.plot(history.history['categorical_accuracy'], label = 'Training precision')
    plt.plot(history.history['val_categorical_accuracy'], label = 'validation precision')
    plt.axhline(0.9,color="#228b22") # see where 90% accuracy is
                
    plt.axhline(maxtest,color="#daa520")
    plt.text(0,maxtest,str(round(maxtest,2)))
    
    plt.axhline(maxtrain,color="#00bfff")
    plt.text(0,maxtrain,str(round(maxtrain,2)))
    
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('entrainement ' + FLAGS.model + ' sur ' + str(num_classes) + ' classes :')
	
    if FLAGS.save_fig: # do you want to save your figure ?
        plt.savefig(figname)
    plt.show()
    
    if FLAGS.save_weights: # do you want to save your model's weights and dictionnay ?
        path = cur_path + 'src/model_RNN_bis/model_'+str.upper(FLAGS.model)+'/'
        nom_model = 'model_'+FLAGS.model+'.json'
        nom_weights = 'model_'+FLAGS.model+'.h5'
        save_model(path, nom_model, nom_weights, model)
        
        dict_path = cur_path + 'src/model_RNN_bis/model_'+str.upper(FLAGS.model)+'/'
        save_dict(dict_path, 'dictionnaire', word_dictionnary = word_dictionnary)

if __name__ == "__main__":
    __main__()