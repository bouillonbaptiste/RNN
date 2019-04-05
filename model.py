# -*- coding: utf-8 -*-
"""
@author: Baptiste

The goal is to classify multiples sentences into several classes.
Different techniques are tested:
CNN:
	we've been creating our own space of words using word2vec in order to give sens to a sentence
RNNs:
	using different types of RNN to see which oneis the best
"""

from __future__ import print_function, division

import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import datetime

import re

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dropout, Conv1D, SimpleRNN, GRU, LSTM, Conv2D
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam, RMSprop, SGD
from keras.constraints import maxnorm
from keras.layers.convolutional import MaxPooling2D
from keras.models import model_from_json

from functions import *
# hidden secret, secret transformations




flags = tf.app.flags

flags.DEFINE_integer('batch_size',10, "The number of sentence to evaluate in a bag")
flags.DEFINE_integer('epochs_nb',500, "The number of itterations")
flags.DEFINE_boolean('weights_save',False, "save the weights or not")
flags.DEFINE_integer('procede',1, "The dictionnary to use in the model") # trandform classes names into numbers
flags.DEFINE_string('model_type','mlp', "the type of model to use")
flags.DEFINE_boolean('save_fig',False, "save the figure or not")
flags.DEFINE_boolean('show_fig',False, "show the figure or not")
flags.DEFINE_float('gamma_reg',0.01, "regularization parameter")
flags.DEFINE_integer('vector_size',32, "The number of word per sentence in model")

flags.DEFINE_string('matr_path','src/variables/model_word2vec_32.npy', "Embeddings matrix to code data")
flags.DEFINE_string('dict_path','src/variables/dictionnary_word2vec_32.npy', "Dictionnaries to code data")

flags.DEFINE_string('layers','[32,64]','description of number of neurons for each layer')

FLAGS = flags.FLAGS



def model_MLP(max_words,maxlen,num_classes,gamma,layers):
    model = Sequential()
    model.add(Embedding(max_words, layers[0], input_length=maxlen))
    # model.add(Dropout(0.5))
    for ii in layers[1:]:
        model.add(Dense(ii, activation='relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(optimizer=Adam(gamma), loss=custom_loss, metrics=['categorical_accuracy'])
    
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


def model_CNN(dim_vec,num_classes,gamma,layers):
    model = Sequential()
    
    # in: 1,32,32 out: 4,16,16
    model.add(Conv2D(4,(2, 2),input_shape=(FLAGS.vector_size, dim_vec, 1),padding='same', activation='relu',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=None,padding='valid',data_format=None))
    
    # in: 4,16,16 out: 64,8,8
    model.add(Conv2D(8,(2, 2),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=None,padding='valid',data_format=None))
    
    # in: 8,8,8 out: 16,4,4
    model.add(Conv2D(16,(2, 2),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=None,padding='valid',data_format=None))
    
    # in: 16,4,4 out: 32,2,2
    model.add(Conv2D(32,(2, 2),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=None,padding='valid',data_format=None))
    
    # in: 32,2,2 out: 64,1,1
    model.add(Conv2D(64,(2, 2),activation='relu',padding='same',kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=None,padding='valid',data_format=None))
    
    model.add(Flatten())
    for ii in layers:
        model.add(Dense(ii,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dense(num_classes, activation='sigmoid'))
    
    model.compile(optimizer=Adam(gamma), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    
    return model


def save_csv(filename,
             model,
             nb_data,
             nb_epochs,
             batch_size,
             gamma,
             vector_size,
             layers,
             train_score,
             test_score):
    
    
    model_resume = pd.read_csv(filename,header=0,sep=";",encoding="Latin-1").dropna(axis=0)
    
    newline = pd.DataFrame({"name":str(datetime.datetime.now())[:16],
                            "model":str(model),
                            "nb_data":str(nb_data),
                            "nb_epochs":str(nb_epochs),
                            "batch_size":str(batch_size),
                            "gamma":str(gamma),
                            "vector_size":str(vector_size),
                            "layers":str(layers),
                            "train_score":str(train_score),
                            "test_score":str(test_score)},index=[0])
    
    model_resume.append(newline).to_csv(filename,index=False,sep=";",encoding="Latin-1")



def __main__CNN():
    
    #############
    ### PATHS ###
    #############
    
    os.chdir(os.path.dirname(os.path.realpath(__file__))+'/')
    os.chdir("../../")
    cur_path = os.getcwd()+"/"
    
    try:
        T_IN = os.environ["IN_TRAINING"]
    except:
        print("les variables d'environnement n'a pas pu être chargée ...")
        T_IN = cur_path+"/INOUT/IN_TRAINING/"
    
    ##################################
    ### LOAD DATA AND DICTIONARIES ###
    ##################################
    
    try:
        description, procede = load_train_dir(T_IN)
    except:
        print("wrong repertory for IN_TRAINING")
    
    dictionnary = np.load(cur_path + FLAGS.dict_path).item()
    embeddings = np.load(FLAGS.matr_path)
    
    if FLAGS.procede == 1:
        proc_codage = procede_codage
        num_classes = len(np.unique([int(ii) for ii in procede_codage.values()]))
    elif FLAGS.procede == 2:
        proc_codage = procede_codage_2
        num_classes = len(np.unique([int(ii) for ii in procede_codage_2.values()]))
    elif FLAGS.procede == 3:
        proc_codage = procede_codage_3
        num_classes = len(np.unique([int(ii) for ii in procede_codage_3.values()]))
    
    dim_vec = FLAGS.vector_size
    
    X = transform_X_CNN(description,dictionnary,embeddings,dim_vec)
    Y = transform_Y(procede,proc_codage)
    
    nb_data = np.shape(X)[0]
    
    X, Y = shuffle_mano(X,Y)
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.2)
    
    log_dir = cur_path+'src/models/model_CNN/logs/'
    
    expr1 = re.compile("[\[|\]|,\"]+")
    layers = [int(ii) for ii in expr1.sub(" ",FLAGS.layers).strip().split(" ")]
    
    # Create the model
    print("using CNN")
    model = model_CNN(dim_vec,num_classes,FLAGS.gamma_reg,layers)
    
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
    
    filename = cur_path+"src/models/model_resume.csv"
    save_csv(filename,
             str(FLAGS.model_type),
             str(nb_data),
             str(FLAGS.epochs_nb),
             str(FLAGS.batch_size),
             str(FLAGS.gamma_reg),
             str(FLAGS.vector_size),
             str(FLAGS.layers),
             str(round(train_score[1],4)),
             str(round(test_score[1],4)))
    
    if FLAGS.show_fig:
        figname = cur_path + 'src/sorties_graphiques/model_' + FLAGS.model_type + '_' + str(FLAGS.epochs_nb) + 'epochs_' + str(FLAGS.procede) + 'proc.png'
        
        maxtrain = max(history.history['categorical_accuracy'])
        maxtest = max(history.history['val_categorical_accuracy'])
        plt.plot(history.history['categorical_accuracy'], label = 'Training precision')
        plt.plot(history.history['val_categorical_accuracy'], label = 'validation precision')
        plt.axhline(0.8,color="#228b22")
                    
        plt.axhline(maxtest,color="#daa520")
        plt.text(0,maxtest,str(round(maxtest,2)))
        
        plt.axhline(maxtrain,color="#00bfff")
        plt.text(0,maxtrain,str(round(maxtrain,2)))
        
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('entrainement ' + FLAGS.model_type + ' sur ' + str(num_classes) + ' classes :')
        if FLAGS.save_fig:
            plt.savefig(figname)
        plt.show()
    
    if FLAGS.weights_save:
        try:
            path = cur_path + 'src/models/model_'+str.upper(FLAGS.model_type)+'/'
            nom_model = 'model_'+FLAGS.model_type+'.json'
            nom_weights = 'model_'+FLAGS.model_type+'.h5'
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
        T_IN = os.environ["IN_TRAINING"]
    except:
        print("variable d'environnement n'a pas pu être chargée ...")
        T_IN = cur_path+"/INOUT/IN_TRAINING/"
    
    #################
    ### LOAD DATA ###
    #################
    
    try:
        description, procede = load_train_dir(T_IN)
    except:
        print("wrong repertory for IN_TRAINING")
    
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
    
    max_length = FLAGS.vector_size
    
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
    
    # predictions = model.predict_classes(X_test)
    
    
    train_score = model.evaluate(X_train, Y_train, batch_size=FLAGS.batch_size)
    test_score = model.evaluate(X_test, Y_test,batch_size=FLAGS.batch_size)
    print('Train accuracy:', train_score[1])
    print('Test accuracy:', test_score[1])
    
    
    filename = cur_path+"src/models/model_resume.csv"
    save_csv(filename,
             str(FLAGS.model_type),
             str(nb_data),
             str(FLAGS.epochs_nb),
             str(FLAGS.batch_size),
             str(FLAGS.gamma_reg),
             str(FLAGS.vector_size),
             str(FLAGS.layers),
             str(round(train_score[1],4)),
             str(round(test_score[1],4)))
    
    
    if FLAGS.show_fig:
        figname = cur_path + 'src/sorties_graphiques/model_' + FLAGS.model_type + '_' + str(FLAGS.epochs_nb) + 'epochs_' + str(FLAGS.procede) + 'proc.png'
        
        maxtrain = max(history.history['categorical_accuracy'])
        maxtest = max(history.history['val_categorical_accuracy'])
        plt.plot(history.history['categorical_accuracy'], label = 'Training precision')
        plt.plot(history.history['val_categorical_accuracy'], label = 'validation precision')
        plt.axhline(0.8,color="#228b22")
                    
        plt.axhline(maxtest,color="#daa520")
        plt.text(0,maxtest,str(round(maxtest,2)))
        
        plt.axhline(maxtrain,color="#00bfff")
        plt.text(0,maxtrain,str(round(maxtrain,2)))
        
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('entrainement ' + FLAGS.model_type + ' sur ' + str(num_classes) + ' classes :')
        if FLAGS.save_fig:
            plt.savefig(figname)
        plt.show()
    
    if FLAGS.weights_save:
        try:
            path = cur_path + 'src/models/model_'+str.upper(FLAGS.model_type)+'/'
            nom_model = 'model_'+FLAGS.model_type+'.json'
            nom_weights = 'model_'+FLAGS.model_type+'.h5'
            save_model(path, nom_model, nom_weights, model)
            
            dict_path = cur_path + 'src/models/model_'+str.upper(FLAGS.model_type)+'/'
            save_dict(dict_path, 'dictionnaire', word_dictionnary = word_dictionnary)
        except:
            print("ERROR: could not save model")


if __name__ == "__main__":
    if FLAGS.model_type=="cnn":
        __main__CNN()
    else:
        __main__RNN()
