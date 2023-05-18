# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:45:13 2018

@author: 13327
"""
from openpyxl import load_workbook
from openpyxl import Workbook
from numpy import argsort
from sklearn.ensemble import RandomForestRegressor
import random
import operator
import numpy as np
import datetime
def range2rect(x,y,start=0):
    M=[]
    N=[]
    for i in range(x):
        for j in range(y):
            N.append(start)
        M.append(N)
        N=[]
    return M
#生成一个M行N列的全为零的矩阵
A=range2rect(831,541)
SM_miRNA=np.loadtxt('SM-miRNA数字关联.txt',dtype=np.int)
for i in SM_miRNA:
    A[i[0]-1][i[1]-1]=1
SM_sim=np.loadtxt('SM similarity matrix.txt')
miRNA_sim=np.loadtxt('miRNA smility maritx.txt')
sm_bianhao=np.loadtxt('SM 编号.txt',str)
miRNA_bianhao=np.loadtxt('miRNA 编号.txt',str)
U_number=[]
for i in range(831):
    for j in range(541):
        if A[i][j]==0:
            U_number.append([i+1,j+1])


X_train=[]
T_lable=[]
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
P=[]
for i in range(664):
    P.append(SM_miRNA[i])
U_all=U_number
random_number=random.sample(range(0,len(U_all)),len(P))
U=[]
for i in random_number:
    U.append(U_all[i])
Fp=[]
for i in range(1372):
    Fp.append(0)
Fp=np.array(Fp)
for i in P:
    featurevector=np.r_[SM_sim[i[0]-1],miRNA_sim[i[1]-1]]
    featurevector=(featurevector-min(featurevector))/(max(featurevector)-min(featurevector))
    Fp=Fp+featurevector
Fn=[]
for i in range(1372):
    Fn.append(0)
Fn=np.array(Fn)
for i in U:
    featurevector=np.r_[SM_sim[i[0]-1],miRNA_sim[i[1]-1]]
    featurevector=(featurevector-min(featurevector))/(max(featurevector)-min(featurevector))
    Fn=Fn+featurevector   
FFp=(Fp-min(Fp))/(max(Fp)-min(Fp))  
FFn=(Fn-min(Fn))/(max(Fn)-min(Fn))
FFn[np.where(FFn==0)]=np.min(FFn[np.where(FFn!=0)])
FScore=FFp/FFn
FScore_=[]
for i in range(len(FScore)):
    FScore_.append([i+1,FScore[i]])

FScore_831=FScore_[:831]
FScore_541=FScore_[831:]
FScore_831.sort(key=operator.itemgetter(1))
FScore_541.sort(key=operator.itemgetter(1))
chose_feature=[]
for i in range(30):
    chose_feature.append(FScore_831[i][0])
for i in range(801,831):
    chose_feature.append(FScore_831[i][0])
for i in range(20):
    chose_feature.append(FScore_541[i][0])
for i in range(521,541):
    chose_feature.append(FScore_541[i][0])

F_V=range2rect(831,541)
for i in range(831):
    for j in range(541):
        vector=np.r_[SM_sim[i],miRNA_sim[j]]
        max_num=max(vector)
        min_num=min(vector)
        vector=(vector-min_num)/(max_num-min_num)

        vector_=[]
        for k in chose_feature:
            vector_.append(vector[k-1])

        F_V[i][j]=vector_
   
X_train=[]
T_lable=[]
for i in P:
    X_train.append(F_V[i[0]-1][i[1]-1])
    T_lable.append(1)
for i in U:
    X_train.append(F_V[i[0]-1][i[1]-1])
    T_lable.append(0)

#开始训练
rf=RandomForestRegressor(n_estimators=100,max_features=0.2,min_samples_leaf=10)
rf.fit(X_train[:],T_lable[:])    

global_list=[]
pr=[]
sample_all=[]
for i in range(831):
    for j in range(541):
        if A[i][j]==0:
            sample_all.append(F_V[i][j])
pr=rf.predict(sample_all)
pr_jiangxu=argsort(-pr)
sample_all=[]
pr_bianhao=[]
x=0
for i in range(831):
    for j in range(541):
        if A[i][j]==0:
            pr_bianhao.append([i,j,pr[x]])
            x+=1
wb=Workbook()
ws=wb.active
ws.title='所有SM的预测结果'
for i in pr_jiangxu:
    ws.append([sm_bianhao[pr_bianhao[i][0]-1][0],miRNA_bianhao[pr_bianhao[i][1]-1][0],pr_bianhao[i][2]])
            
wb.save(filename='所有SM的预测结果.xlsx')
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            
            

            
            
            
            
            
            
            
            
            
            
