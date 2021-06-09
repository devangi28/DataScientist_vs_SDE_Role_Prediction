#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 03:09:01 2021

@author: junghopark
"""

from scipy import sparse
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import os

"""Import vectors from saved directory"""
def import_vectors(path):
    X_train_vector = sparse.load_npz('{}/data/vecdata/X_train_vec.npz'.format(path))
    X_test_vector = sparse.load_npz('{}/data/vecdata/X_test_vec.npz'.format(path))
    Y_train_vector = np.load('{}/data/vecdata/Y_train_vec.npy'.format(path))
    Y_test_vector = np.load('{}/data/vecdata/Y_test_vec.npy'.format(path))
    
    return X_train_vector, X_test_vector, Y_train_vector, Y_test_vector

"""Perform Gridsearch to find best parameters - KNN, LREG, DT"""    
def gridsearch(KNN, LREG, DT, X_train_vector, Y_train_vector):
    
    print('----Gridsearch start----')
    
    #KNN gridsearch
    KNN_grid =  [{'n_neighbors': [1,3,5,7,9,11,13,15,17], 'weights':['uniform','distance']}]
    #build a grid search to find the best parameters
    gridsearchKNN = GridSearchCV(KNN, KNN_grid, cv=5, verbose=True) 
    #run gridsearch
    gridsearchKNN.fit(X_train_vector,Y_train_vector)
    
    #print best parameter and score - KNN
    KNN_best_params, KNN_best_score = gridsearchKNN.best_params_, gridsearchKNN.best_score_
    print('KNN best parameters: ', KNN_best_params)
    print('KNN best score: ', KNN_best_score)
    
    #Logistic Regression gridsearch
    LREG_grid = [ {'C':[0.5,1,1.5,2],'penalty':['l1','l2']}]
    #build a gridsearch to find the best parameters
    gridsearchLREG = GridSearchCV(LREG, LREG_grid, cv=5, verbose=True)
    #run gridsearch
    gridsearchLREG.fit(X_train_vector, Y_train_vector)
    
    #print best parameter and score - LREG
    LREG_best_params, LREG_best_score = gridsearchLREG.best_params_, gridsearchLREG.best_score_
    print('LREG best parameters ', LREG_best_params)
    print('LREG best score', LREG_best_score)
    
    #Decision tree gridsearch
    DT_grid = [{'max_depth': [3,4,5,6,7,8,9,10,11,12],'criterion':['gini','entropy']}]
    #build a gridsearch to find best params
    gridsearchDT = GridSearchCV(DT, DT_grid, cv=5, verbose=True)
    #run gridsearch
    gridsearchDT.fit(X_train_vector, Y_train_vector)
    
    DT_best_params, DT_best_score = gridsearchDT.best_params_, gridsearchDT.best_score_
    print('DT best parameters ', DT_best_params)
    print('DT best score', DT_best_score)
    
    print('----Gridsearch end----')
    
    return KNN_best_params, LREG_best_params, DT_best_params

"""Do a voting classifier among 3 models"""
def voting_classifier(KNN_classifier, LREG_classifier, DT_classifier):
    #do voting classifier
    predictors=[('knn',KNN_classifier),('lreg',LREG_classifier),('dt',DT_classifier)]
    VT=VotingClassifier(predictors)
    VT.fit(X_train_vector,Y_train_vector)
    
    #use voting classifier for prediction
    predicted=VT.predict(X_test_vector).tolist()
    
    return predicted #return prediction

"""Execute main function"""
if __name__ == '__main__':
    #directory path for vector files
    path = '/C:/Users/devan/OneDrive/Desktop/Web Mining/Team4_BIA660_FinalProject_Submission'
    
    X_train_vector, X_test_vector, Y_train_vector, Y_test_vector = import_vectors(path)
    
    #define binary classification models
    KNN_classifier=KNeighborsClassifier()
    LREG_classifier=LogisticRegression(solver='liblinear')
    DT_classifier = DecisionTreeClassifier()
    
    #perform gridsearch and store best parameters
    KNN_best_params, LREG_best_params, DT_best_params = gridsearch(KNN_classifier, LREG_classifier, DT_classifier, X_train_vector, Y_train_vector)
    
    Y_prediction = voting_classifier(KNN_classifier, LREG_classifier, DT_classifier)
    
    #confusion matrix
    cm = confusion_matrix(Y_prediction, Y_test_vector)
    
    #compare results
    print('---- Accuracy Score ----')
    print(accuracy_score(Y_prediction,Y_test_vector))
    print('---- Confusion Matrix ----')
    print(cm)
    
    #make result folder
    if not os.path.exists('{}/results'.format(path)):
        os.makedirs('{}/results'.format(path))    
    
    #plot confusion matrix and save as png
    df_cm = pd.DataFrame(cm, index = ['True', 'False'], columns =['True', 'False'])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('{}/results/confusion_matrix.png'.format(path))
        
    #save final prediction results with tests
    actual_pred = list(zip(Y_test_vector,Y_prediction))
    df_result = pd.DataFrame(actual_pred ,columns=['Actual job title','Predicted job title'])
    df_result.to_csv('{}/results/results.csv'.format(path), index = False)
    
