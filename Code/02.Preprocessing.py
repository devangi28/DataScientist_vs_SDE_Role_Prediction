#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 01:20:06 2021

@author: junghopark
"""

import glob
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
from scipy import sparse


"""Merge scraped data into one big csv file, headers of latter files are excluded"""
def merge_csv(path):
    file_count=0
    want_header = True
    #combined csv file name
    out_filename = "combined_jobs.csv"
    
    #if file exists, delete to rewrite
    if os.path.exists('{}/'.format(path)+out_filename):
        os.remove('{}/'.format(path)+out_filename)
    
    read_files = glob.glob("{}/*.csv".format(path))

    #merge every file in data folder
    with open('{}/'.format(path) + out_filename, "w") as outfile:
        for filename in read_files:
            file_count += 1
            with open(filename) as infile:
                if want_header:
                    outfile.write('{}\n'.format(next(infile).strip()).lower())
                    want_header = False
                else:
                    next(infile)
                for line in infile:
                    outfile.write('{}\n'.format(line.strip().lower()))
    
    print("---- {} files were merged ----".format(file_count))
    
    return outfile

"""Change csv to df, drop duplicates, get all X and all Y and return """
def load_csv(filepath):
    combined_jobs_df = pd.read_csv(filepath)
    #drop duplicates
    print('Total merged data entries count: {}'.format(str(len(combined_jobs_df))))
    print('Job counts: \n{}'.format(combined_jobs_df['jobtitle'].value_counts()))
    combined_jobs_df = combined_jobs_df.drop_duplicates()
    
    all_X = combined_jobs_df['jobdesc']
    all_Y = combined_jobs_df['jobtitle']
    
    return all_X, all_Y

"""Count vectorize train data"""
def vectorize_job_desc_data(X_train, X_test):
    
    #define counter
    counter = CountVectorizer(stop_words=stopwords.words('english'))
    counter.fit(X_train)
    
    #count the number of times each term appears in a document and transform each doc into a count vector
    X_train_vectorized = counter.transform(X_train)
    X_test_vectorized = counter.transform(X_test)
    
    return X_train_vectorized, X_test_vectorized

"""Vectorize job title {'data scientist:0','software engineer:1'}"""
def vectorize_job_title_data(Y_train, Y_test):
    Y_train_vectorized = []
    for row in Y_train:
        if row == "data scientist":
            Y_train_vectorized.append(0)
        elif row == "software engineer":
            Y_train_vectorized.append(1)

    Y_test_vectorized = []
    for row in Y_test:
        if row == "data scientist":
            Y_test_vectorized.append(0)
        elif row == "software engineer":
            Y_test_vectorized.append(1)

    return Y_train_vectorized, Y_test_vectorized

"""Save vectors into /vecdata folder"""
def save_vectors(X_train_vectorized, X_test_vectorized, Y_train_vectorized, Y_test_vectorized, path):
    if not os.path.exists('{}/vecdata'.format(path)):
        os.makedirs('{}/vecdata'.format(path))
    sparse.save_npz("{}/vecdata/X_train_vec.npz".format(path), X_train_vectorized)
    sparse.save_npz("{}/vecdata/X_test_vec.npz".format(path), X_test_vectorized)
    np.save("{}/vecdata/Y_train_vec.npy".format(path), Y_train_vectorized)
    np.save("{}/vecdata/Y_test_vec.npy".format(path), Y_test_vectorized)

"""Execute main function to see counts for distinct variables and store vectors for training & testing"""
if __name__ == '__main__':
    #path of the data folder / change it for desired path and where data belongs
    path = '/Users/junghopark/Desktop/Stevens_Coursework/Spring_2021/BIA 660 Web mining/Final Project/data'
    
    #execute merge
    combined_jobs_csv = merge_csv(path)
    
    #load merged csv for preprocessing
    with open('{}/combined_jobs.csv'.format(path), encoding="utf8") as csvfile:
        all_X, all_Y = load_csv(csvfile)
    
    #show description of x & y variables
    print('----X variable description----')
    print(all_X.describe())
    print('----Y variable description----')
    print(all_Y.describe())
    #print(all_Y[all_Y['jobtitle'] == 'data scientist'].count())
    
    #create new csv with no duplicates
    #combined_no_duplicates.to_csv('combined_jobs_no_duplicates.csv')
    
    #Split data into train test
    X_train, X_test, Y_train, Y_test = train_test_split(all_X, all_Y, test_size = 0.2, random_state = 0)
    
    #Vectorize words in job description
    X_train_vectorized, X_test_vectorized = vectorize_job_desc_data(X_train, X_test)
    
    #Vectorize job title
    Y_train_vectorized, Y_test_vectorized = vectorize_job_title_data(Y_train, Y_test)
    
    #Save vectors in to given path to be ready for training & testing
    save_vectors(X_train_vectorized, X_test_vectorized, Y_train_vectorized, Y_test_vectorized, path)
    
    print('----train,test data split----')
    print('train size : {}'.format(len(X_train)))
    print('test size : {}'.format(len(X_test)))
    print('----train,test data count----')
    print('Data Scientist in training set: ', Y_train_vectorized.count(0))
    print('Data Scientist in test set: ', Y_test_vectorized.count(0))
    print('Software Engineer in training set: ', Y_train_vectorized.count(1))
    print('Software Engineer in test set: ', Y_test_vectorized.count(1))