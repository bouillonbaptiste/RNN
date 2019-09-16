# -*- coding: utf-8 -*-
"""
@author: Baptiste

The goal is to classify multiples sentences into several classes.
Different techniques are tested:
CNN:
	we have create our own space of words using word2vec in order to give sens to a sentence
RNNs:
	using different types of RNN to see which one is the best

This script include only the creation of a model.
Scripts containing the training of an existing model, evaluation of sentences using a model,
summary of all models created, embedding space creation, ... are not revealed.
"""

from __future__ import print_function, division

try:
    from functions import *
    print("functions imported")
except:
    print("could not import functions ...")
# secret transformations


import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from datetime import datetime

import re

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dropout, Conv1D, SimpleRNN, GRU, LSTM, Conv2D, Input
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D
from keras.layers import Concatenate, MaxPool2D, Reshape

from sklearn.model_selection import train_test_split

from keras.layers.convolutional import MaxPooling2D

from keras.models import Model
from keras.models import model_from_json

from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.constraints import maxnorm







flags = tf.app.flags

flags.DEFINE_integer('batch_size',50, "The number of sentence to evaluate in a bag")
flags.DEFINE_integer('epochs_nb',50, "The number of word per sentence in model")
flags.DEFINE_boolean('weights_save',False, "save the weigths or not")
flags.DEFINE_integer('procede',1, "The procede_codage to use in the model")
flags.DEFINE_string('model_type','mlp', "the type of model to use")
flags.DEFINE_boolean('show_fig',False, "save the figure or not")
flags.DEFINE_float('gamma_reg',0.01, "regularization parameter")
flags.DEFINE_integer('words_nb',32, "The number of word per sentence in model")
flags.DEFINE_integer('embed_dim',32, "The number of dimensions of the embeddings model")
flags.DEFINE_string('matrice_name',"","the path to cost matrix for the loss function")
flags.DEFINE_string('lot_id','TOTAL','The batch to be trained on')

flags.DEFINE_string('matr_path','src/variables/model_word2vec_32_TATIA.npy', "Embeddings matrix to code data")
flags.DEFINE_string('dict_path','src/variables/dictionnary_word2vec_32_TATIA.npy', "Dictionnaries to code data")

flags.DEFINE_string('layers','[16,32]','description of number of neurons each layer')

FLAGS = flags.FLAGS



def model_MLP(max_words,maxlen,num_classes,gamma,layers):
    model = Sequential()
    model.add(Embedding(max_words, layers[0], input_length=maxlen))
    # model.add(Dropout(0.5))
    for ii in layers[1:]:
        model.add(Dense(ii, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(num_classes, activation='sigmoid'))
    #model.compile(optimizer=Adam(gamma), loss=custom_loss2, metrics=['categorical_accuracy'])
    model.compile(optimizer=Adam(gamma), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    
    return model

def model_RNN(max_words,maxlen,num_classes,gamma,layers):
    model = Sequential()
    model.add(Embedding(max_words, layers[0], input_length=maxlen))
    # model.add(Dropout(0.5))
    for ii in layers[1:]:
        model.add(SimpleRNN(ii, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=RMSprop(gamma), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    
    return model

def model_LSTM(max_words,maxlen,num_classes,gamma,layers):
    model = Sequential()
    model.add(Embedding(max_words, layers[0], input_length=maxlen))
    # model.add(Dropout(0.5))
    for ii in layers[1:]:
        model.add(LSTM(ii, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=RMSprop(gamma), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model

def model_GRU(max_words,maxlen,num_classes,gamma,layers):
    model = Sequential()
    model.add(Embedding(max_words, 32, input_length=maxlen))
    # model.add(Dropout(0.5))
    for ii in layers[1:]:
        model.add(GRU(ii, activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=Adam(gamma), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model


def model_CNN(words_nb=32,embed_dim=32,num_classes=53,gamma_reg=0.05,layers=[32,32]):
    
    # NEW #####################################################################
    
    inputs = Input(shape=(embed_dim, words_nb, 1))
    reshape = Reshape((embed_dim,words_nb,1))(inputs)
    
    
    conv_0 = Conv2D(64, kernel_size=(embed_dim, 2), padding='same', kernel_initializer='normal', activation='relu')(reshape)
    conv_1 = Conv2D(64, kernel_size=(embed_dim, 4), padding='same', kernel_initializer='normal', activation='relu')(reshape)
    #conv_2 = Conv2D(128, kernel_size=(embed_dim, 8), padding='same', kernel_initializer='normal', activation='relu')(reshape)
    
    maxpool_0 = MaxPool2D(pool_size=(words_nb - 2 + 1, 1), strides=(2,2), padding='same')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(words_nb - 4 + 1, 1), strides=(2,2), padding='same')(conv_1)
    #maxpool_2 = MaxPool2D(pool_size=(words_nb - 8 + 1, 1), strides=(2,2), padding='same')(conv_2)
    
    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1])#, maxpool_2])
    flatten = Flatten()(concatenated_tensor)
    #flatten = Flatten()(maxpool_0)
    dropout = Dropout(0.1)(flatten)
    layerii = Dense(layers[0],activation='relu')(dropout)
    for ii in layers[1:]:
        layerii = Dense(ii,activation='relu')(layerii)
    output = Dense(units=num_classes, activation='softmax')(layerii)
    
    # this creates a model that includes
    model = Model(inputs=inputs, outputs=output)
    
    
    model.compile(optimizer=Adam(gamma_reg), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def save_csv(filename,
             name,
             lot_id,
             model_type,
             nb_data,
             epochs_nb,
             batch_size,
             gamma_reg,
             words_nb,
             layers,
             train_score,
             test_score):
    
    model_resume = pd.read_csv(filename,header=0,sep=";",encoding="Latin-1").dropna(axis=0)
    
    newline = pd.DataFrame({"name":str(name),
                            'lot_id':str(lot_id),
                            "model_type":str(model_type),
                            "nb_data":str(nb_data),
                            "epochs_nb":str(epochs_nb),
                            "batch_size":str(batch_size),
                            "gamma_reg":str(gamma_reg),
                            "words_nb":str(words_nb),
                            "layers":str(layers),
                            "train_score":str(train_score),
                            "test_score":str(test_score)},index=[0])
    
    model_resume.append(newline,sort=False,).to_csv(filename,index=False,sep=";",encoding="Latin-1")



def __main__CNN():
    
    #############
    ### PATHS ###
    #############
    
    os.chdir(os.path.dirname(os.path.realpath(__file__))+'/')
    os.chdir("../../")
    cur_path = os.getcwd()+"/"
    try:
        TATIA_IN = '{0}/INOUT/TATIA_IN_TRAINING/LOTS/{1}/'.format(cur_path,FLAGS.lot_id)
    except:
        print("ENV variables could not be loaded ...")
        TATIA_IN = '{0}/INOUT/TATIA_IN_TRAINING/LOTS/{1}/'.format(cur_path,FLAGS.lot_id)
    
    ##################################
    ### LOAD DATA AND DICTIONARIES ###
    ##################################
    
    try:
        description, procede = load_train_dir(TATIA_IN)
    except:
        print('wrong repertory for TATIA_IN_TRAINING')
    
    dictionnary = np.load(cur_path + FLAGS.dict_path,allow_pickle=True).item()
    embeddings = np.load(cur_path + FLAGS.matr_path,allow_pickle=True)
    
    if FLAGS.procede == 1:
        proc_codage = procede_codage
        num_classes = len(np.unique([int(ii) for ii in procede_codage.values()]))
    elif FLAGS.procede == 2:
        proc_codage = procede_codage_2
        num_classes = len(np.unique([int(ii) for ii in procede_codage_2.values()]))
    elif FLAGS.procede == 3:
        proc_codage = procede_codage_3
        num_classes = len(np.unique([int(ii) for ii in procede_codage_3.values()]))
    
    dim_vec = FLAGS.words_nb
    
    X = transform_X_CNN(description,dictionnary,embeddings,dim_vec)
    Y = transform_Y(procede,proc_codage)
    
    
    nb_data = np.shape(X)[0]
    
    X, Y = shuffle_mano(X,Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.2)
    
    log_dir = '{}INOUT/MODELS/model_CNN/logs/'.format(cur_path)
    
    expr1 = re.compile("[\[|\]|,\"]+")
    layers = [int(ii) for ii in expr1.sub(" ",FLAGS.layers).strip().split(" ")]
    
    # Create the model
    print("using CNN")
    model = model_CNN(FLAGS.words_nb,FLAGS.embed_dim,num_classes,FLAGS.gamma_reg,layers)
    
    callbacks = []
    
    history = model.fit(X_train,
                        y_train,
                        epochs = FLAGS.epochs_nb,
                        batch_size = FLAGS.batch_size,
                        shuffle = True,
                        callbacks = callbacks)
    
    train_score = model.evaluate(X_train, y_train, batch_size=FLAGS.batch_size, verbose=1)
    test_score = model.evaluate(X_test, y_test,batch_size=FLAGS.batch_size, verbose=1)
    print('Train accuracy:', train_score[1])
    print('Test accuracy:', test_score[1])
    
    date_name = str(datetime.now())
    
    filename = cur_path+"INOUT/MODELS/model_resume.csv"
    save_csv(filename,
             str(date_name),
             str(FLAGS.lot_id),
             str(FLAGS.model_type),
             str(nb_data),
             str(FLAGS.epochs_nb),
             str(FLAGS.batch_size),
             str(FLAGS.gamma_reg),
             str(FLAGS.words_nb),
             str(FLAGS.layers),
             str(round(train_score[1],4)),
             str(round(test_score[1],4)))
    
    
    path2 = cur_path
    if cur_path[:9]=='/home/ia/':
        path2 = '/home/tatia/'
    figname = '{0}INOUT/MODELS/sorties_graphiques/{1}.png'.format(path2,date_name.replace(' ','_'))
    
    maxtrain = max(history.history['acc'])
    
    f, plot = plt.subplots()
    plot.plot(history.history['acc'], label = 'Training precision')
    plt.axhline(maxtrain,color="#00bfff")
    plt.text(0,maxtrain,str(round(maxtrain,2)))
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('entrainement {0} sur {1} classes :'.format(FLAGS.model_type,str(num_classes)))
    plt.savefig(figname)
    if FLAGS.show_fig:
        plt.show()
    
    if FLAGS.weights_save:
        try:
            path = '{0}INOUT/MODELS/model_{1}/'.format(cur_path,str.upper(FLAGS.model_type))
            nom_model = 'model_{}.json'.format(FLAGS.model_type)
            nom_weights = 'model_{}.h5'.format(FLAGS.model_type)
            save_model(path, nom_model, nom_weights, model)
        except:
            print("ERROR: could not save model")

def __main__RNN():
    
    #############
    ### PATHS ###
    #############
    
    os.chdir(os.path.dirname(os.path.realpath(__file__))+'/')
    os.chdir("../../")
    cur_path = os.getcwd()+"/"
    
    try:
        TATIA_IN = '{0}/INOUT/TATIA_IN_TRAINING/LOTS/{1}/'.format(cur_path,FLAGS.lot_id)
    except:
        print("ENV variables could not be loaded ...")
        TATIA_IN = '{0}/INOUT/TATIA_IN_TRAINING/LOTS/{1}/'.format(cur_path,FLAGS.lot_id)
    
    #################
    ### LOAD DATA ###
    #################
    
    try:
        description, procede = load_train_dir(TATIA_IN)
    except:
        print("wrong repertory for TATIA_IN_TRAINING")
    
    word_dictionnary = dict_word(description)
    
    if FLAGS.procede == 1:
        proc_codage = procede_codage
        num_classes = len(np.unique([int(ii) for ii in procede_codage.values()]))
    elif FLAGS.procede == 2:
        proc_codage = procede_codage_2
        num_classes = len(np.unique([int(ii) for ii in procede_codage_2.values()]))
    elif FLAGS.procede == 3:
        proc_codage = procede_codage_3
        num_classes = len(np.unique([int(ii) for ii in procede_codage_3.values()]))
    
    max_length = FLAGS.words_nb
    
    X = transform_X_RNN(description,word_dictionnary,max_length)
    Y = transform_Y(procede, proc_codage)
    
    nb_data = np.shape(X)[0]
    
    X, Y = shuffle_mano(X,Y)
    
    
    max_words = len(word_dictionnary.values())
    gamma = FLAGS.gamma_reg
    
    labelsnb = range(0,len(proc_codage.keys())+1)
    class_weight = {}
    for index, label in enumerate(labelsnb):
        class_weight[index] = len([x for x in procede if proc_codage[x]==label])/len(Y)
    
    expr1 = re.compile("[\[|\]|,\"]+")
    layers = [int(ii) for ii in expr1.sub(" ",FLAGS.layers).strip().split(" ")]
    
    if FLAGS.model_type == "mlp":
        print("using MLP")
        model = model_MLP(max_words,max_length,num_classes,gamma,layers)
    elif FLAGS.model_type == "rnn":
        print("using RNN")
        model = model_RNN(max_words,max_length,num_classes,gamma,layers)
    elif FLAGS.model_type == "lstm":
        print("using LSTM")
        model = model_LSTM(max_words,max_length,num_classes,gamma,layers)
    elif FLAGS.model_type == "gru":
        print("using GRU")
        model = model_GRU(max_words,max_length,num_classes,gamma,layers)
    
    
    callbacks = []
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    
    history = model.fit(X_train, Y_train,
                        class_weight=class_weight,
                        epochs=FLAGS.epochs_nb,
                        batch_size=FLAGS.batch_size,
                        validation_split=0.2,
                        callbacks=callbacks,
                        initial_epoch=0)
    
    
    train_score = model.evaluate(X_train, Y_train, batch_size=FLAGS.batch_size)
    test_score = model.evaluate(X_test, Y_test,batch_size=FLAGS.batch_size)
    print('Train accuracy:', train_score[1])
    print('Test accuracy:', test_score[1])
    
    date_name = str(datetime.now())
    
    filename ='{}"INOUT/MODELS/model_resume.csv'.format(cur_path)
    save_csv(filename,
             str(date_name),
             str(FLAGS.lot_id),
             str(FLAGS.model_type),
             str(nb_data),
             str(FLAGS.epochs_nb),
             str(FLAGS.batch_size),
             str(FLAGS.gamma_reg),
             str(FLAGS.words_nb),
             str(FLAGS.layers),
             str(round(train_score[1],4)),
             str(round(test_score[1],4)))
    
    
    path2 = cur_path
    if cur_path[:9]=='/home/ia/':
        path2 = '/home/tatia/'
    figname = '{0}INOUT/MODELS/sorties_graphiques/{1}.png'.format(path2,date_name.replace(' ','_'))
    
    maxtrain = max(history.history['categorical_accuracy'])
    maxtest = max(history.history['val_categorical_accuracy'])
    
    f, plot = plt.subplots()
    plot.plot(history.history['categorical_accuracy'], label = 'Training precision')
    plt.plot(history.history['val_categorical_accuracy'], label = 'validation precision')
    plt.axhline(0.8,color="#228b22")
                
    plt.axhline(maxtest,color="#daa520")
    plt.text(0,maxtest,str(round(maxtest,2)))
    
    plt.axhline(maxtrain,color="#00bfff")
    plt.text(0,maxtrain,str(round(maxtrain,2)))
    
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('entrainement {0} sur {1} classes :'.format(FLAGS.model_type,str(num_classes)))
    plt.savefig(figname)
    if FLAGS.show_fig:
        plt.show()
    
    if FLAGS.weights_save:
        try:
            path = '{0}INOUT/MODELS/model_{1}/'.format(cur_path,str.upper(FLAGS.model_type))
            nom_model = 'model_{}.json'.format(FLAGS.model_type)
            nom_weights = 'model_{}.h5'.format(FLAGS.model_type)
            save_model(path, nom_model, nom_weights, model)
            dict_path = '{0}INOUT/MODELS/model_{1}/'.format(cur_path,str.upper(FLAGS.model_type))
            save_dict(dict_path, 'dictionnaire', word_dictionnary = word_dictionnary)
        except:
            print("ERROR: could not save model")


if __name__ == "__main__":
    if FLAGS.model_type=="cnn":
        __main__CNN()
    else:
        __main__RNN()
