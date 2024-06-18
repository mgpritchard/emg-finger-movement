#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:18:33 2024

@author: michael
"""
from sklearn.feature_selection import SelectPercentile, f_classif, SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np


def sel_feats_l1_df(data,sparsityC=0.01,maxfeats=None):
    target=data['Label']
    attribs=data.drop(columns=['Label'])   
    lsvc = LinearSVC(C=sparsityC, penalty="l1", dual=False).fit(attribs, target)
    if maxfeats is None:
        #getting all nonzero feats
        model = SelectFromModel(lsvc, prefit=True)
    else:
        #getting feats up to maxfeats according to coefs, with no threshold of coef, so as to ensure it never returns <maxfeats
        model = SelectFromModel(lsvc, prefit=True,threshold=-np.inf,max_features=maxfeats)   
    col_idxs=model.get_support(indices=True)
    return col_idxs

def scale_feats_train(data,mode='normalise'):
    '''data is a dataframe of feats, mode = normalise or standardise'''
    if mode is None:
        return data, None
    if mode=='normalise' or mode=='normalize':
        scaler=Normalizer()
    elif mode=='standardise' or mode=='standardize':
        scaler=StandardScaler()
    cols_to_ignore=list(data.filter(regex='^ID_').keys())
    cols_to_ignore.append('Label')
    data[data.columns[~data.columns.isin(cols_to_ignore)]]=scaler.fit_transform(data[data.columns[~data.columns.isin(cols_to_ignore)]])
    return data, scaler

def scale_feats_test(data,scaler):
    '''data is a dataframe of feats, scaler is a scaler fit to training data'''
    if scaler is None:
        return data
    cols_to_ignore=list(data.filter(regex='^ID_').keys())
    cols_to_ignore.append('Label')
    #data[data.columns[~data.columns.isin(cols_to_ignore)]]=scaler.fit_transform(data[data.columns[~data.columns.isin(cols_to_ignore)]])
    data[data.columns[~data.columns.isin(cols_to_ignore)]]=scaler.transform(data[data.columns[~data.columns.isin(cols_to_ignore)]])
    # also pandas SettingWithCopyWarning would suggest this doesnt actually affect data, but it demonstrably does
    # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
    return data

def drop_ID_cols(csv_dframe):
    IDs=csv_dframe.filter(regex='^ID_').columns
    csv_dframe=csv_dframe.drop(IDs,axis='columns')
    '''may benefit from a trycatch in case of keyerror?'''
    return csv_dframe

