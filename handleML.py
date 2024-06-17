#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 19:17:17 2024

@author: michael
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def train_optimise(training_set,modeltype,args,bagging=False):
    '''where training_set is a Pandas dataframe
    which has Label as the last column but has had ID columns dropped'''
    if modeltype=='RF':
        model=train_RF_param(training_set,args)
    elif modeltype=='LDA':
        model=train_LDA_param(training_set,args)
    elif modeltype=='SVM':
        model = train_SVC_Platt(training_set,args)
    else:
        raise ValueError(f"Unidentified model type: {modeltype}!")
    return model

def train_SVC_Platt(train_data,args):
    svmargs=args['svmArgs']
    kernel=svmargs['kernel_type']
    C=svmargs['svm_C']
    gamma=svmargs['gamma']
    if kernel=='linear':
        model=SVC(C=C,kernel=kernel,probability=True) #possible need to fix random_state as predict is called multiple times?
    else:
        svc=SVC(C=C,kernel=kernel,gamma=gamma)
        model=CalibratedClassifierCV(svc,cv=5)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_LDA_param(train_data,args):
    solverArgs=args['LDA_solver']
    solver=solverArgs['solver_type']
    if solver == 'svd':
        model=LinearDiscriminantAnalysis(solver=solver)
    else:
        shrinkage=solverArgs['shrinkage']
        model=LinearDiscriminantAnalysis(solver=solver,shrinkage=shrinkage)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def train_RF_param(train_data,args):
    '''where args is a dictionary with n_trees as an integer item within'''
    n_trees=args['n_trees']
    max_depth=args['max_depth']
    model=RandomForestClassifier(n_estimators=n_trees,max_depth=max_depth)
    train=train_data.values[:,:-1]
    targets=train_data.values[:,-1]
    model.fit(train.astype(np.float64),targets)
    return model

def prob_dist(model,values):
    distro = model.predict_proba(values)
    distro[distro==0]=0.00001
    return distro

def predict_from_array(model,values):
	prediction = model.predict(values)
	return prediction

def pred_from_distro(labels,distro):
    pred=int(np.argmax(distro))
    label=labels[pred]
    return label

def predlist_from_distrosarr(labels,distros):
    predcols=np.argmax(distros,axis=1)
    predlabels=labels[predcols]
    return predlabels.tolist()