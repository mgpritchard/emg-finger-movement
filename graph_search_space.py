#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 18:43:16 2024

@author: michael
"""

import re
from collections.abc import Iterable
import graphviz
####https://github.com/opizzato/dict_to_digraph/blob/main/dict_to_digraph.py

import os
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin/"





emg=graphviz.Digraph('emg',edge_attr={'arrowhead':'none'},node_attr={'shape':'box','style':'rounded'})
emg.node('alg','Classification Algorithm')

emg.node('alg.svm','SVM',style='solid')
emg.node('alg.svm.kernel','Kernel: RBF',style='solid')
emg.node('alg.svm.rbf.c','C')
emg.node('alg.svm.rbf.c.vals','0.1 - 100.0\n(log scaled)',style='solid')
emg.node('alg.svm.rbf.gamma','gamma')
emg.node('alg.svm.rbf.gamma.vals','0.01 - 1.0\n(log scaled)',style='solid')

emg.node('alg.lda','LDA',style='solid')
emg.node('alg.lda.solver','Solver')
emg.node('alg.lda.svd','Singular Value\nDecomposition',style='solid')
emg.node('alg.lda.eigen','Eigenvalue\nDecomposition',style='solid')
emg.node('alg.lda.eigen.shrinkage','Shrinkage')
emg.node('alg.lda.eigen.shrinkage.vals','0.0 - 1.0',style='solid')
emg.node('alg.lda.lsqr','Least Squares\nSolution',style='solid')
emg.node('alg.lda.lsqr.shrinkage','Shrinkage')
emg.node('alg.lda.lsqr.shrinkage.vals','0.0 - 1.0',style='solid')

emg.node('alg.rf','RF',style='solid')
emg.node('alg.rf.ntrees','# Trees')
emg.node('alg.rf.ntrees.vals','10 - 100\n(steps of 5)',style='solid')
emg.node('alg.rf.maxdepth','Max Tree\nDepth')
emg.node('alg.rf.maxdepth.vals','2 - 5',style='solid')



emg.edges([('alg','alg.svm'),('alg','alg.lda'),('alg','alg.rf'),
               ('alg.svm','alg.svm.kernel'),('alg.svm','alg.svm.rbf.c'),('alg.svm','alg.svm.rbf.gamma'),
               ('alg.svm.rbf.c','alg.svm.rbf.c.vals'),('alg.svm.rbf.gamma','alg.svm.rbf.gamma.vals'),
               ('alg.lda','alg.lda.solver'),
               ('alg.lda.solver','alg.lda.svd'),('alg.lda.solver','alg.lda.eigen'),('alg.lda.solver','alg.lda.lsqr'),
               ('alg.lda.eigen','alg.lda.eigen.shrinkage'),('alg.lda.eigen.shrinkage','alg.lda.eigen.shrinkage.vals'),
               ('alg.lda.lsqr','alg.lda.lsqr.shrinkage'),('alg.lda.lsqr.shrinkage','alg.lda.lsqr.shrinkage.vals'),
               ('alg.rf','alg.rf.ntrees'),('alg.rf','alg.rf.maxdepth'),
               ('alg.rf.ntrees','alg.rf.ntrees.vals'),('alg.rf.maxdepth','alg.rf.maxdepth.vals'),
              ])

emg.render(view=True)






