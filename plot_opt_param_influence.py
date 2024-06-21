# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:34:23 2023

@author: pritcham
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import developModel as modelling
import scipy.stats as stats

def boxplot_param(df_in,param,target,ylower=0,yupper=1,showplot=True,xlabs=None,title=None,titleheight=0.98,xlabel=None,ylabel=None):
    fig,ax=plt.subplots()
    dataframe=df_in.copy()
    if isinstance(dataframe[param][0],list):
        dataframe[param]=dataframe[param].apply(lambda x: x[0])
    dataframe.boxplot(column=target,by=param,ax=ax,showmeans=True)
    ax.set_ylim(ylower,yupper)
    if xlabs is not None:
        #ax.set_xticks(np.arange(1,len(xlabs)+1),xlabs)
        ax.set_xticklabels(xlabs)
   # plt.suptitle('')
    ax.set_title('')
    if title is not None:
    #    ax.set_title(title)
        plt.suptitle(title,y=titleheight)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if showplot:
        plt.show()
    return fig

def model_significance(df,target,param,winner,title='Mean accuracy per classifier in Bespoke EEG-Only optimisation',titleheight=0.98,models=None):
    if models is None:
        models = ['RF','KNN','LDA','QDA','GNB','SVM']
    model_dict=dict(zip(range(0,len(models)+1),models))
    
    per_param=boxplot_param(df,param,target,xlabs=models,
                               title=title,titleheight=titleheight)
    
    stat_test=df[[param,target]]
    stat_test[param]=[x[0] for x in stat_test[param]]
    stat_test[param]=[model_dict[x] for x in stat_test[param]]
    groups = [stat_test[stat_test[param] == group][target] for group in models]
    #sorting the above by bespoke_models instead of #stat_test['eeg model'].unique()] #so that order is preserved
    fstat, pvalue = stats.f_oneway(*groups)
    #print('anova on all ',fstat, pvalue)
    #https://saturncloud.io/blog/anova-in-python-using-pandas-dataframe-with-statsmodels-or-scipy/#:~:text=ANOVA%20is%20a%20fundamental%20statistical,popular%20libraries%3A%20Statsmodels%20and%20Scipy.
    
    return per_param



def LDA_solver_plot(df,modelparam,resultparam,solverparam,trial='',titleheight=0.98,solvers=None):
    if solvers is None:
        solvers=['svd','lsqr','eigen']
    
    df_subset=df[[modelparam,resultparam,solverparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==1].reset_index(drop=True)
    
    per_LDAsolver=boxplot_param(df_subset,solverparam,resultparam,xlabs=solvers,titleheight=0.995,
                                xlabel='Solver',ylabel='Mean classification accuracy',
                               title=trial+'\nMean accuracy per LDA solver')
    
    if 0:
        solver_groups = [df_subset[df_subset[solverparam] == group][resultparam] for group in [[0],[2]]]
        fstat, pvalue = stats.f_oneway(*solver_groups)
        print('anova on all ',fstat, pvalue)
        per_LDAsolver_stats=boxplot_param(df_subset,solverparam,resultparam,xlabs=solvers,titleheight=0.995,
                                    xlabel='Solver',ylabel='Mean classification accuracy',
                                   title=trial+'\nMean accuracy per LDA solver, ANOVA: f='+str(round(fstat,4))+', p'+(('='+str(round(pvalue,4))) if round(pvalue,4)!=0 else '<0.0001'))
        
    return per_LDAsolver
          

def SVM_params_significance(df,modelparam,resultparam,Cparam,Gammaparam,trial=''):
    df_subset=df[[modelparam,resultparam,Cparam,Gammaparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==5].reset_index(drop=True)
    df_subset[Cparam]=[x[0] for x in df_subset[Cparam]]
    df_subset[Gammaparam]=[x[0] for x in df_subset[Gammaparam]]
    
    pearsonR=stats.pearsonr(df_subset[Cparam],df_subset[resultparam])
    spearmanR=stats.spearmanr(df_subset[Cparam],df_subset[resultparam])
    df_subset.plot(x=Cparam,y=resultparam,kind='scatter')#,logx=True)#ylim=(0.8,1),
    title=(trial+'\nAccuracy vs C, Pearson coefficient = '+str(round(pearsonR[0],4))+' ('+('p='+str(min(1.0,round(pearsonR[1]*2,4))) if round(pearsonR[1]*2,4)!=0 else 'p'+'<0.0001')+')'
           +'\n'+f"{' '*32}"+' Spearman\'s rho = '+str(round(spearmanR[0],4))+' ('+('p='+str(min(1.0,round(spearmanR[1]*2,4))) if round(spearmanR[1]*2,4)!=0 else 'p'+'<0.0001')+')')
    plt.gcf().suptitle(title,y=1.025) #0.995 for two-line
    plt.xlabel('C')
    plt.ylabel('Mean classification accuracy')
    
    pearsonR=stats.pearsonr(df_subset[Gammaparam],df_subset[resultparam])
    spearmanR=stats.spearmanr(df_subset[Gammaparam],df_subset[resultparam])
    df_subset.plot(x=Gammaparam,y=resultparam,kind='scatter')#,logx=True)
    #title=trial+'\nAccuracy vs Gamma, pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4)))
    #title=trial+'\nAccuracy vs Gamma: (Pearson coefficient, p) = '+(str((round(pearsonR[0],4),round(pearsonR[1],4))) if round(pearsonR[1],4)!=0 else '('+str(round(pearsonR[0],4))+', <0.0001)')
    title=(trial+'\nAccuracy vs Gamma, Pearson coefficient = '+str(round(pearsonR[0],4))+' ('+('p='+str(min(1.0,round(pearsonR[1]*2,4))) if round(pearsonR[1]*2,4)!=0 else 'p'+'<0.0001')+')'
           +'\n'+f"{' '*36}"+' Spearman\'s rho = '+str(round(spearmanR[0],4))+' ('+('p='+str(min(1.0,round(spearmanR[1]*2,4))) if round(spearmanR[1]*2,4)!=0 else 'p'+'<0.0001')+')')
    plt.gcf().suptitle(title,y=1.025)
    plt.xlabel('Gamma')
    plt.ylabel('Mean classification accuracy')
    
  #  df_subset['invert']=1-df_subset[resultparam]
  #  df_subset.plot(x=Gammaparam,y='invert',kind='scatter',loglog=True,ylim=(0,1),
  #                      title='Accuracy vs Gamma, pearson coefficient = '+str((round(pearsonR[0],4),round(pearsonR[1],4))))


    
def RF_trees_significance(df,modelparam,resultparam,Treesparam,trial=None):
    df_subset=df[[modelparam,resultparam,Treesparam]]
    df_subset[modelparam]=[x[0] for x in df_subset[modelparam]]
    df_subset=df_subset.loc[df_subset[modelparam]==0].reset_index(drop=True)
    df_subset[Treesparam]=[x[0] for x in df_subset[Treesparam]]
    
    pearsonR=stats.pearsonr(df_subset[Treesparam],df_subset[resultparam])
    #arguably just monotonic because its technically categorical?? but no.
    spearmanR=stats.spearmanr(df_subset[Treesparam],df_subset[resultparam])
    
    title=(trial+'\nAccuracy vs # of trees, Pearson coefficient = '+str(round(pearsonR[0],4))+' ('+('p='+str(round(pearsonR[1]*2,4)) if round(pearsonR[1]*2,4)!=0 else 'p'+'<0.0001')+')'
           +'\n'+f"{' '*36}"+' Spearman\'s rho = '+str(round(spearmanR[0],4))+' ('+('p='+str(round(spearmanR[1]*2,4)) if round(spearmanR[1]*2,4)!=0 else 'p'+'<0.0001')+')')
    df_subset.plot(x=Treesparam,y=resultparam,kind='scatter')
    plt.gcf().suptitle(title,y=1.025)
    plt.xlabel('Number of trees')
    plt.ylabel('Mean classification accuracy')

if __name__=='__main__':

    plt.rcParams['figure.dpi']=150
    
    print('BONFERRONI FOR ALL CASES WHERE BOTH R AND RHO, P IS DOUBLED')
    
    
    respath = '/home/michael/Documents/Aston/PostSubmission/RATask/results_10iter_backup/trials_obj.pkl'
    
    _,_,opt_results=modelling.load_results_obj(respath)
    
    models=['RF','LDA','SVM']
    
    model_fig=model_significance(opt_results,'emg_acc','emg model','LDA',title='Mean accuracy per classifier in EMG optimisation',models=models)
    
    solver_fig=LDA_solver_plot(opt_results,'emg model','emg_acc','emg.lda.solver',trial='LDAs in EMG System Optimisation',solvers=['svd','eigen'])
    
    #raise
    
    RF_trees_significance(opt_results, 'emg model', 'emg_acc', 'emg.rf.ntrees',trial='RFs in EMG System Optimisation')
    
    SVM_params_significance(opt_results,'emg model','emg_acc','emg.svm.rbf.c','emg.svm.rbf.gamma',trial='SVMs in EMG System Optimisation')
    
    
  
