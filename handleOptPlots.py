#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:28:55 2024

@author: michael
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_stat_in_time(trials,stat,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    ax.plot(range(1, len(trials) + 1),
            [x['result'][stat] for x in trials], 
        color='red', marker='.', linewidth=0)
    #https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt?scriptVersionId=12981074&cellId=97
    ax.set(title=stat+' over optimisation iterations')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

def plot_multiple_stats(trials,stats,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    for stat in stats:
        ax.plot(range(1, len(trials) + 1),
                [x['result'][stat] for x in trials],
                label=(stat))
    #ax.set(title=stat+' over optimisation iterations')
    ax.legend()#loc='upper center')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

def calc_runningbest(trials,stat=None):
    if stat is None:
        best=np.maximum.accumulate([1-x['result']['loss'] for x in trials])
    else:
        best=np.maximum.accumulate([x['result'][stat] for x in trials])
    return best

def plot_multiple_stats_with_best(trials,stats,runbest=None,ylower=0,yupper=1,showplot=True):
    if isinstance(trials,pd.DataFrame):
        fig=plot_multi_runbest_df(trials,stats,runbest,ylower,yupper,showplot)
        return fig
    fig,ax=plt.subplots()
    for stat in stats:
        ax.plot(range(1, len(trials) + 1),
                [x['result'][stat] for x in trials],
                label=(stat))
        #https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt?scriptVersionId=12981074&cellId=97
    #ax.set(title=stat+' over optimisation iterations')
    best=calc_runningbest(trials,runbest)
    ax.plot(range(1,len(trials)+1),best,label='running best')
    ax.legend()#loc='upper center')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

def plot_multi_runbest_df(trials,stats,runbest,ylower,yupper,showplot):
    fig,ax=plt.subplots()
    for stat in stats:
        trials[stat].plot(ax=ax,label=stat)
     #   ax.plot(range(1, len(trials) + 1),
      #          [trials[stat].T],
       #         label=(stat))
        #https://www.kaggle.com/code/fanvacoolt/tutorial-on-hyperopt?scriptVersionId=12981074&cellId=97
    #ax.set(title=stat+' over optimisation iterations')
    if runbest is not None:
        best=np.fmax.accumulate(trials[runbest])
        best.plot(ax=ax,label='running best')
    ax.legend()#loc='upper center')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig

def boxplot_param(df_in,param,target,ylower=0,yupper=1,showplot=True):
    fig,ax=plt.subplots()
    dataframe=df_in.copy()
    if isinstance(dataframe[param][0],list):
        dataframe[param]=dataframe[param].apply(lambda x: x[0])
    dataframe.boxplot(column=target,by=param,ax=ax,showmeans=True)
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    return fig
  
def scatterbox(trials,stat='fusion_accs',ylower=0,yupper=1,showplot=True):
    ''' likely unused, mainly to show distribution across ppts'''
    fig,ax=plt.subplots()
    X=range(1, len(trials) + 1)
    H=[x['result'][stat] for x in trials]
    groups = [[] for i in range(max(X))]
    [groups[X[i]-1].append(H[i]) for i in range(len(H))]
    groups=[each[0] for each in groups]
    ax.boxplot(groups,showmeans=True)
    ax.set(title=stat+' over optimisation iterations')
    ax.set_ylim(ylower,yupper)
    if showplot:
        plt.show()
    #https://stackoverflow.com/questions/53473733/how-to-add-box-plot-to-scatter-data-in-matplotlib
    return fig