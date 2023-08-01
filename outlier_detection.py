import numpy as np
import pandas as pd
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.kpca import KPCA
from pyod.utils.utility import standardizer
from pyod.models.combination import  average
from pyod.utils.utility import argmaxn

data=pd.read_excel('final_QSAR.xlsx')
x=np.array(data.iloc[:,-1]).reshape(-1,1)

detector=[KNN(),IForest(),HBOS(),CBLOF(),KPCA()]
x_scores=np.zeros([len(x),len(detector)])

for i in range(len(detector)):
    clf=detector[i]
    clf.fit(x)
    x_scores[:, i]=clf.decision_scores_ # raw outlier scores

x_scores_norm= standardizer(x_scores)

comb_by_average = average(x_scores_norm)

data['outliers']=np.zeros([len(x)])

data['outliers'][argmaxn(comb_by_average, n=14)]=1

data.to_excel('QSAR_refined.xlsx')
