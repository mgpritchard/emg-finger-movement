#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 00:10:07 2024

@author: michael
"""

import pickle
import handleFeats as feats
import handleML as ml
import time
import numpy as np
import pandas as pd
import developModel as modelling


def validate_candidate(args):
    start=time.time()
    if not args['data_in_memory']:
        emg_train=pd.read_csv(args['emg_training_path'],delimiter=',')
        emg_validate=pd.read_csv(args['emg_validation_path'])
    else:
        emg_train=args['emg_train']
        emg_validate=args['emg_validate']
   # if not args['prebalanced']: 
   #     emg_ppt=balance_set(emg_ppt)
    
    
    emg_train.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    emg_train=emg_train.reset_index(drop=True)
    
    emg_validate.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    emg_validate=emg_validate.reset_index(drop=True)
      
    if args['scalingtype']:
        emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
        emg_validate=feats.scale_feats_test(emg_validate,emgscaler)

    if args['get_train_acc']:
        emg_trainacc=emg_train.copy()
   
    emg_train=emg_train.reset_index(drop=True)
    
    emg_train=feats.drop_ID_cols(emg_train)
    
    sel_cols_emg=feats.sel_feats_l1_df(emg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
    sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
    emg_train=emg_train.iloc[:,sel_cols_emg]
    
    emg_model = ml.train_optimise(emg_train, args['emg']['emg_model_type'], args['emg'])
    classlabels = emg_model.classes_
    
    emg_validate.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        
    targets, predlist_emg, distros_emg = modelling.predict_set(emg_validate, emg_model, classlabels, args, sel_cols_emg, get_distros=True)


    #gest_truth,gest_pred_emg,gesturelabels = classes_from_preds(targets,predlist_emg,classlabels)
    gest_truth,gest_pred_emg,gesturelabels = targets,predlist_emg,classlabels
 
    emg_acc = modelling.accuracy_score(gest_truth,gest_pred_emg)
    kappa = modelling.cohen_kappa_score(gest_truth,gest_pred_emg)
    

    if args['get_train_acc']:
        traintargs, predlist_train, distros_train = modelling.predict_set(emg_trainacc, emg_model, classlabels, args, sel_cols_emg, get_distros=True)
        #train_truth=[idx_to_gestures[gest] for gest in traintargs]
        train_truth=traintargs
        #train_preds=[idx_to_gestures[pred] for pred in predlist_train]
        train_preds=predlist_train
        train_acc = modelling.accuracy_score(train_truth,train_preds)
    else:
        train_acc = 0
        
    if args['plot_confmats']:
       # gesturelabels=[idx_to_gestures[label] for label in classlabels]
        gesturelabels=classlabels
        gesturelabels=['thumb','open','victory','middle','ring','little','rest']
        modelling.confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
    
    end=time.time()
    return {
        'loss': 1-emg_acc,
        'status': 'STATUS_OK',
        'kappa':kappa,
        'emg_acc':emg_acc,
        'train_acc':train_acc,
        'targets':targets,
        'predictions':predlist_emg,
        'pred_distros':distros_emg,
        'elapsed_time':end-start,}


if __name__ == '__main__':
    
    identified_params_path='/home/michael/Documents/Aston/PostSubmission/RATask/results/winner_params.pkl'
    
    candidate_params=pickle.load(open(identified_params_path,'rb'))
    
    candidate_params.update({'emg_training_path':'/home/michael/Documents/Aston/PostSubmission/RATask/working-dataset/featureset/traintestFeats_Labelled.csv',
                            'emg_validation_path':'/home/michael/Documents/Aston/PostSubmission/RATask/working-dataset/featureset/sparetestFeats_Labelled.csv',
                            'data_in_memory':False,
                            'plot_confmats':True,
                            'get_train_acc':True,
                            'get_distros':True})
    
    results=validate_candidate(candidate_params)