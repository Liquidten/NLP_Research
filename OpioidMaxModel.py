#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:51:02 2018

@author: sameepshah
"""

import glob as glob
import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
#from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling1D
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn import metrics




positive_Test = glob.glob("/Users/sameepshah/Desktop/Data/Opioid_Data/Data/Test/Yes/*.txt")
negative_Test = glob.glob("/Users/sameepshah/Desktop/Data/Opioid_Data/Data/Test/No/*.txt")
positive_Train = glob.glob("/Users/sameepshah/Desktop/Data/Opioid_Data/Data/Train/Yes/*.txt")
negative_Train = glob.glob("/Users/sameepshah/Desktop/Data/Opioid_Data/Data/Train/No/*.txt")

def read_files_in_one_dataframe_column(file_name_list):
	result_df_list = []
	for file_name in file_name_list:
		result_df = pd.read_csv(file_name, names=["Cuis"])
		result_df_list.append(result_df)

	sum_result_df = pd.concat(result_df_list)
	
	return 	sum_result_df


df_Test_P = read_files_in_one_dataframe_column(positive_Test)
#print(df_Test_P)
df_Test_N = read_files_in_one_dataframe_column(negative_Test)
df_Train_P = read_files_in_one_dataframe_column(positive_Train)
df_Train_N = read_files_in_one_dataframe_column(negative_Train)


positive_Test_D = pd.DataFrame(df_Test_P)
positive_Test_D["label"] = 0

negative_Test_D = pd.DataFrame(df_Test_N)
negative_Test_D["label"] = 1

Test = pd.concat(objs = [positive_Test_D, negative_Test_D],
					axis = 0,
					join = "outer")

positive_Train_D = pd.DataFrame(df_Train_P)
positive_Train_D["label"] = 0

negative_Train_D = pd.DataFrame(df_Train_N)
negative_Train_D["label"] = 1


Train = pd.concat(objs = [positive_Train_D, negative_Train_D],
					axis = 0,
					join = "outer")

Test.columns = ['Cuis','labels']
Train.columns = ['Cuis','labels']
Test_file = (Test['Cuis'])
Test_label = Test['labels']
Train_file = (Train['Cuis'])
Train_label = Train['labels']



def textprocessing(Train_file, Train_label,Test_file, Test_label, MAXLEN):
    X_file, X_dev, y_label, y_dev = train_test_split(Train_file, Train_label, test_size = 0.15, random_state=0)
    tokenizer = Tokenizer(lower = False, oov_token='UNK')
    tokenizer.fit_on_texts(X_file)
    vocab_size = len(tokenizer.word_index) + 1
    #print(encoded)# determine the vocabulary size
    print('Vocabulary Size: %d' % vocab_size)
    encoded_train = tokenizer.texts_to_sequences(X_file)
    X_train = pad_sequences(encoded_train, maxlen = MAXLEN, padding='post')
    
    encoded_dev = tokenizer.texts_to_sequences(X_dev)
    X_dev = pad_sequences(encoded_dev, maxlen = MAXLEN, padding='post')
    #print(X_train.shape)
    
    encoded_test = tokenizer.texts_to_sequences(Test_file)
    X_test = pad_sequences(encoded_test, maxlen = MAXLEN, padding='post')
    #print(X_test.shape)
    
    return X_train, X_dev, X_test, y_label, y_dev, Test_label, vocab_size
    

def maxlength(Train_file):
    CUIsLength = []
    for x in Train_file:
        y = x.split()
        z = len(y)
        #print(z)
        CUIsLength.append(z)
    highest = max(CUIsLength)
    #print(highest)
    return highest


if __name__=="__main__":
    
    Maxlen = maxlength(Train_file)
    #print("vocabSize: " + str(vocabSize))
    print("Maxlen:" + str(Maxlen))
    
    X_train, X_dev, X_test, y_label, y_dev, Test_label, vocab_size = textprocessing(Train_file, Train_label,Test_file, Test_label, Maxlen) 
    print(X_train.shape)
    print(y_label.shape)
    Embedding_layer = Embedding(vocab_size,150,input_length=Maxlen)
    
    model = Sequential()
    model.add(Embedding_layer)
    model.add(GlobalMaxPooling1D())
    model.add(Dropout(0.541))
    
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    #sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',optimizer=Adam ,metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir="logs/run_a")
    x = model.fit(np.array(X_train), np.array(y_label), epochs=15, batch_size = 5, validation_data = (np.array(X_dev), np.array(y_dev)), callbacks=[tensorboard])
    prediction = model.predict_classes(np.array(X_test), batch_size = 1)
    print(prediction)
    accuracy = accuracy_score(np.array(Test_label), prediction)
    print(accuracy)
    
    prediction2 = model.predict_proba(np.array(X_test), batch_size = 1)
    pct_auc = metrics.roc_auc_score(Test_label,prediction) * 100
    print('{:0.2}'.format(pct_auc))
    
    
    
    loss = x.history['loss']
    val_loss = x.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
    plt.title('Traning and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    plt.clf()
    acc = x.history['acc']
    val_acc = x.history['val_acc']
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
    plt.title('Traning and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()
    
    
    
    
    