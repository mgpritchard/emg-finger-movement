#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 17:22:44 2024

@author: michael
"""
import sys
import os
import numpy as np
import pandas as pd
#import statistics as stats
from hyperopt import fmin, tpe, hp, space_eval, STATUS_OK, Trials
from hyperopt.pyll import scope, stochastic
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay #plot_confusion_matrix
import matplotlib.pyplot as plt
import handleFeats as feats
import handleML as ml
from generate_training_matrix import idx_to_gestures, gestures_to_idx
import handleOptPlots as optplot
import time
import random
import pickle

def setup_search_space():
    emgoptions=[
                {'emg_model_type':'RF',
                 'n_trees':scope.int(hp.quniform('emg.rf.ntrees',10,100,q=5)),
                 'max_depth':scope.int(hp.quniform('emg.rf.maxdepth',2,5,q=1)),
                 #'max_samples':scope.int(hp.quniform('emg.RF.maxsamples',))
                 #default max samples is length of train data.
                 #could search over eg half length up to length, but would need
                 #length to be passed in or hardcoded
                 },
                {'emg_model_type':'LDA',
                 'LDA_solver':hp.choice('emg.lda.solver',[
                     {'solver_type':'svd'},
                     {'solver_type':'lsqr',
                      'shrinkage':hp.uniform('emg.lda.lsqr.shrinkage',0.0,1.0)},
                     {'solver_type':'eigen',
                      'shrinkage':hp.uniform('emg.lda.eigen.shrinkage',0.0,1.0)},
                     ]),
                 },
                {'emg_model_type':'SVM',
                 'svmArgs':hp.choice('emg.svm.kernel',[
                     {'kernel_type':'rbf',
                      'svm_C':hp.loguniform('emg.svm.rbf.c',np.log(0.1),np.log(100)),
                      'gamma':hp.loguniform('emg.svm.rbf.gamma',np.log(0.01),np.log(1))},
                   #  {'kernel_type':'linear',
                   #   'svm_C':hp.loguniform('emg.svm.linear.c',np.log(0.1),np.log(100))},
                     ]),                 
                 },
                ]

    space = {
            'emg':hp.choice('emg model',emgoptions),

            'emg_set_path':'/home/michael/Documents/Aston/PostSubmission/RATask/working-dataset/featureset/traintestFeats_Labelled.csv',
            # can get to "here" with os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            'using_literature_data':True,
            'data_in_memory':False,
            'prebalanced':False,
            'scalingtype':'standardise',
            #'scalingtype':hp.choice('scaling',['normalise','standardise']),#,None]),
            'plot_confmats':False,
            'get_train_acc':True,
            }
   
    return space

def balance_set(emg_set):
    emg_set.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)    
    emg=emg_set.reset_index(drop=True)
    
    stratsize=np.min(emg['Label'].value_counts())
    print('subsampling to ',str(stratsize),' per class')
    balemg = emg.groupby('Label',group_keys=False)
    balemg=balemg.apply(lambda x: x.sample(stratsize))
    
    return balemg


def classes_from_preds(targets,predlist_emg,classlabels):
    '''Convert predictions to gesture labels'''
    gest_truth=[idx_to_gestures[gest] for gest in targets]
    gest_pred_emg=[idx_to_gestures[pred] for pred in predlist_emg]
    gesturelabels=[idx_to_gestures[label] for label in classlabels]
    return gest_truth,gest_pred_emg,gesturelabels


def confmat(y_true,y_pred,labels,modelname="",testset="",title=""):
    '''y_true = actual classes, y_pred = predicted classes,
    labels = names of class labels'''
    conf=confusion_matrix(y_true,y_pred,labels=labels,normalize='true')
    cm=ConfusionMatrixDisplay(conf,labels)
    #cm=ConfusionMatrixDisplay.from_predictions(y_true,y_pred,labels,normalise=None) #only in skl 1.2
    if modelname != "" and testset != "":
        title=modelname+'\n'+testset
    fig,ax=plt.subplots()
    ax.set_title(title)
    cm.plot(ax=ax)
    #return conf


def predict_set(test_set_emg,model_emg,classlabels,args, chosencolsemg=None, get_distros=False):
    predlist_emg=[]
    targets=[]
    
    emg=test_set_emg.reset_index(drop=True)
    
    targets=emg['Label'].values.tolist()
        
    '''Get values from instances'''
    IDs=list(emg.filter(regex='^ID_').keys())
    emg=emg.drop(IDs,axis='columns')
    
    if chosencolsemg is not None:
        emg=emg.iloc[:,chosencolsemg]
    emgvals=emg.drop(['Label'],axis='columns').values

    '''Pass values to model'''
    distros_emg=ml.prob_dist(model_emg,emgvals)
    predlist_emg=ml.predlist_from_distrosarr(classlabels,distros_emg)

    if get_distros:
        return targets, predlist_emg, distros_emg
    else:
        return targets, predlist_emg, None
    


def train_test(args):
    start=time.time()
    if not args['data_in_memory']:
        emg_set_path=args['emg_set_path']
        emg_ppt=pd.read_csv(emg_set_path,delimiter=',')
    else:
        emg_ppt=args['emg_set']
    if not args['prebalanced']: 
        emg_ppt=balance_set(emg_ppt)
    
    
    emg_ppt.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
    emg_ppt=emg_ppt.reset_index(drop=True)
    emg_ppt['ID_stratID']=emg_ppt['ID_run'].astype(str)+emg_ppt['Label'].astype(str)+emg_ppt['ID_gestrep'].astype(str)

    random_split=random.randint(0,100)
    gest_perfs=emg_ppt['ID_stratID'].unique()
    gest_strat=pd.DataFrame([gest_perfs,[perf.split('.')[1][-1] for perf in gest_perfs]]).transpose()
    train_split,test_split=train_test_split(gest_strat,test_size=0.33,random_state=random_split,stratify=gest_strat[1])

    emg_train=emg_ppt[emg_ppt['ID_stratID'].isin(train_split[0])]
    emg_test=emg_ppt[emg_ppt['ID_stratID'].isin(test_split[0])]
      
    if args['scalingtype']:
        emg_train,emgscaler=feats.scale_feats_train(emg_train,args['scalingtype'])
        emg_test=feats.scale_feats_test(emg_test,emgscaler)

    if args['get_train_acc']:
        emg_trainacc=emg_train.copy()
   
    emg_train=emg_train.reset_index(drop=True)
    
    emg_train=feats.drop_ID_cols(emg_train)
    
    sel_cols_emg=feats.sel_feats_l1_df(emg_train,sparsityC=args['l1_sparsity'],maxfeats=args['l1_maxfeats'])
    sel_cols_emg=np.append(sel_cols_emg,emg_train.columns.get_loc('Label'))
    emg_train=emg_train.iloc[:,sel_cols_emg]
    
    emg_model = ml.train_optimise(emg_train, args['emg']['emg_model_type'], args['emg'])
    classlabels = emg_model.classes_
    
    emg_test.sort_values(['ID_pptID','ID_run','Label','ID_gestrep','ID_tend'],ascending=[True,True,True,True,True],inplace=True)
        
    targets, predlist_emg,_ = predict_set(emg_test, emg_model, classlabels, args, sel_cols_emg, get_distros=True)
    #gest_truth,gest_pred_emg,gesturelabels = classes_from_preds(targets,predlist_emg,classlabels)
    gest_truth,gest_pred_emg,gesturelabels = targets,predlist_emg,classlabels
 
    emg_acc = accuracy_score(gest_truth,gest_pred_emg)
    kappa = cohen_kappa_score(gest_truth,gest_pred_emg)
    

    if args['get_train_acc']:
        traintargs, predlist_train,_ = predict_set(emg_trainacc, emg_model, classlabels, args, sel_cols_emg, get_distros=True)
        #train_truth=[idx_to_gestures[gest] for gest in traintargs]
        train_truth=traintargs
        #train_preds=[idx_to_gestures[pred] for pred in predlist_train]
        train_preds=predlist_train
        train_acc = accuracy_score(train_truth,train_preds)
    else:
        train_acc = 0
        
    if args['plot_confmats']:
       # gesturelabels=[idx_to_gestures[label] for label in classlabels]
        gesturelabels=classlabels
        confmat(gest_truth,gest_pred_emg,gesturelabels,title='EMG')
    
    end=time.time()
    return {
        'loss': 1-emg_acc,
        'status': STATUS_OK,
        'kappa':kappa,
        'emg_acc':emg_acc,
        'train_acc':train_acc,
        'elapsed_time':end-start,}


def optimise_model(prebalance=False,platform='not server',iters=35):
    space=setup_search_space()
    
    if platform=='server':
        space.update({'emg_set_path':'dummy path'})
        raise ValueError('not set up for remote processing, only here for posterity')
    
    if prebalance:
        emg_set=pd.read_csv(space['emg_set_path'],delimiter=',')
        emg_set=balance_set(emg_set)
        space.update({'emg_set':emg_set,'data_in_memory':True,'prebalanced':True})
    else:
        ''' NOT balancing by default because we know there are fewer of victory_sign'''
        emg_set=pd.read_csv(space['emg_set_path'],delimiter=',')
        space.update({'emg_set':emg_set,'data_in_memory':True,'prebalanced':True})
        
    trials=Trials()

    space.update({'l1_sparsity':0.005}) #0.002
    space.update({'l1_maxfeats':67}) # 67=sqrt(6804*0.66)=sqrt(4490), ie size of train set
    best = fmin(train_test,
            space=space,
            algo=tpe.suggest,
            max_evals=iters,
            trials=trials)
    
    return best, space, trials
    


def save_resultdict(filepath,resultdict,dp=4):
    
    status=resultdict['Results'].pop('status')
    
    f=open(filepath,'w')
    try:
        target=list(resultdict['Results'].keys())[list(resultdict['Results'].values()).index(1-resultdict['Results']['loss'])]
        f.write(f"Optimising for {target}\n\n")
    except ValueError:
        target, _ = min(resultdict['Results'].items(), key=lambda x: abs(1-resultdict['Results']['loss'] - x[1]))
        f.write(f"Probably optimising for {target}\n\n")
    
    if 'emg' in resultdict['Chosen parameters']:
        f.write('EMG Parameters:\n')
        for k in resultdict['Chosen parameters']['emg'].keys():
            if isinstance(resultdict['Chosen parameters']['emg'][k],str):
                f.write(f"\t'{k}':{resultdict['Chosen parameters']['emg'][k]}'\n")
            elif isinstance(resultdict['Chosen parameters']['emg'][k],dict):
                for j in resultdict['Chosen parameters']['emg'][k].keys():
                    if isinstance(resultdict['Chosen parameters']['emg'][k][j],str):
                        f.write(f"\t'{k}':{resultdict['Chosen parameters']['emg'][k][j]}'\n")
                    else:
                        f.write(f"\t'{k}':'{round(resultdict['Chosen parameters']['emg'][k][j],dp)}")
            else:
                f.write(f"\t'{k}':'{round(resultdict['Chosen parameters']['emg'][k],dp)}")
                            
    
    f.write('Results:\n')
    resultdict['Results']['status']=status
    for k in resultdict['Results'].keys():
        f.write(f"\t'{k}':'{round(resultdict['Results'][k],dp)if not isinstance(resultdict['Results'][k],str) else resultdict['Results'][k]}'\n")
    
    f.close()

def load_results_obj(path):
    load_trials=pickle.load(open(path,'rb'))
    load_table=pd.DataFrame(load_trials.trials)
    load_table_readable=pd.concat(
        [pd.DataFrame(load_table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(load_table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
    return load_trials,load_table,load_table_readable





if __name__ == '__main__':

    if len(sys.argv)>1:
        platform=sys.argv[1]
        if platform=='not_server':
            platform='not server'
        if len(sys.argv)>2:
            num_iters=int(sys.argv[2])
        if len(sys.argv)>3:
            showplots=sys.argv[3].lower()
        else:
            showplots=None
    else:
        platform='not server'
        num_iters=10
        showplots=None
        
        
    if (platform=='server') or (showplots=='false'):
        showplot_toggle=False
    else:
        showplot_toggle=True


    best,space,trials=optimise_model(platform=platform,iters=num_iters)
    
        
    if 1:  #if showplots ?  
        chosen_space=space_eval(space,best)
        chosen_space['plot_confmats']=True
        chosen_results=train_test(chosen_space)

    best_results=trials.best_trial['result']
    bestparams=space_eval(space,best)
    
    for static in ['emg_set_path','using_literature_data','data_in_memory']:
        bestparams.pop(static,None)
    bestparams.pop('emg_set')
    #bestparams.drop('emg_feats_LOO',errors='ignore') #if a DF   
    bestparams.pop('emg_feats_LOO',None)
    
    print(bestparams)
    print('Best mean accuracy: ',1-best_results['loss'])
         
    winner={'Chosen parameters':bestparams,
            'Results':best_results}
    
    table=pd.DataFrame(trials.trials)
    table_readable=pd.concat(
        [pd.DataFrame(table['result'].tolist()),
         pd.DataFrame(pd.DataFrame(table['misc'].tolist())['vals'].values.tolist())],
        axis=1,join='outer')
    
    '''SETTING RESULT PATH'''
    currentpath=os.path.dirname(__file__)
    result_dir='../results/'
    resultpath=os.path.join(currentpath,result_dir)

    resultpath='/home/michael/Documents/Aston/PostSubmission/RATask/emg-finger-movement/../results/'    
    
    '''PICKLING THE TRIALS OBJECT'''
    trials_obj_path=os.path.join(resultpath,'trials_obj.pkl')
    pickle.dump(trials,open(trials_obj_path,'wb'))
    
    '''CODE FOR LOADING TRIALS OBJ'''
    #load_trials_var=pickle.load(open(filename,'rb'))
    #load_trials,load_table,load_table_readable=load_results_obj(filepath)
    
    '''saving best parameters & results'''
    reportpath=os.path.join(resultpath,'params_results_report.txt')
    save_resultdict(reportpath,winner)
    
    bestparams_path=os.path.join(resultpath,'winner_params.pkl')
    pickle.dump(bestparams,open(bestparams_path,'wb'))
    
    

    '''plotting performance throughout opt process'''
    emg_acc_plot=optplot.plot_stat_in_time(trials,'emg_acc',showplot=showplot_toggle)
    # BELOW IF NOT REPORTING TRAIN ACCURACY
    #acc_compare_plot=optplot.plot_multiple_stats_with_best(trials,['emg_mean_acc'],runbest='emg_mean_acc',showplot=showplot_toggle)  
    # BELOW IF REPORTING TRAIN ACCURACY
    acc_compare_plot=optplot.plot_multiple_stats_with_best(trials,['emg_acc','train_acc'],runbest='emg_acc',showplot=showplot_toggle)
    
    '''saving figures of performance over time'''
    emg_acc_plot.savefig(os.path.join(resultpath,'emg_acc.png'))
    acc_compare_plot.savefig(os.path.join(resultpath,'emg_acc_compare.png'))
    
    '''plotting accuracy per algorithm'''
    per_emgmodel=optplot.boxplot_param(table_readable,'emg model','emg_acc',showplot=showplot_toggle)
    per_emgmodel.savefig(os.path.join(resultpath,'emg_model.png'))
 