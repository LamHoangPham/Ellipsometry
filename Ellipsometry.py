# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 00:37:03 2020

@author: PHAM
"""

import numpy as np
from random import randrange
import SCE


# 1. Helpful functions
def angle2ST(x,s,t):  
    
    # x: array of (azimuth, ellipsity), len(x) depends on s and t
    # s,t: number of PSG state and PSA state, respectively
    
    # number of parameters=2*(s + t)    
    # case 1: 1 PSG + 4 PSA: 10 parameters
    # case 2: 2 PSG + 2PSA: 8 parameters    
    
    angleS=np.array([[x[i],x[s+t+i]] for i in range(s+t) if i<s])
    angleT=np.array([[x[i],x[s+t+i]] for i in range(s+t) if i>=s])
    
    vectorS=Fvector(angleS)
    vectorT=Fvector(angleT)
    
    return vectorS, vectorT   


def Fvector(angle):
    
    def vectori(alpi,epsi): # vector element
        alphai=np.radians(alpi) # covert to radians
        epsiloni=np.radians(epsi)
        return np.array([[1,np.cos(2*alphai)*np.cos(2*epsiloni),np.sin(2*alphai)*np.cos(2*epsiloni),np.sin(2*epsiloni)]]).T
    
    vector=vectori(angle.item(0,0),angle.item(0,1)) # initializing vector
    if len(angle) >1:
        for i in range(1,len(angle)):
            s1=angle.item(i,0)
            s2=angle.item(i,1)
            vector=np.append(vector,vectori(s1,s2),axis=1)   
           
    return vector

# Define function to optimize

def roundmatrix(M): # round all element of matrix
    for i in range (0,M.shape[0]):
        for j in range(0, M.shape[1]):
            M.itemset((i,j), np.round(M.item(i,j),5))    
            
           
#2. Start to optimizatize 
            
def Ellipsometry(OptFunction,number,s,t):
    
    #OptFunction: function for the optimization
    # number: number of run (with different x0)
    
    # s,t: number of PSG state and PSA state, respectively, defining the dimension of starting point x0   
    
    # parameters for SCEUA optimazation
    maxn=100000
    kstop=1000
    pcento=1.e-5
    peps=0.001
    iseed= 0
    iniflg=0
    ngs=5

    # bound for each variable
    pi=180
    bl=np.concatenate(([-pi/2]*(s+t),[-pi/4]*(s+t)),axis=0)
    bu=np.concatenate(([pi/2]*(s+t),[pi/4]*(s+t)),axis=0)


    # run optimization with number values of  x0

    Listbestx=[];Listbestf=[]
    List_vectorS=[];List_vectorT=[]

    for i in range(number):
        #starting with a random x0
        x0=np.concatenate(([randrange(-90,90,1) for i in range(s+t)],[randrange(-45,45,1) for i in range(s+t)]),axis=0)

        bestx,bestf,BESTX,BESTF,ICALL = SCE.sceua(x0,bl,bu,maxn,kstop,pcento,peps,ngs,iseed,iniflg,OptFunction)

        Listbestx.append(bestx)
        Listbestf.append(bestf)
    
        S,T=angle2ST(bestx,s,t)
    
        roundmatrix(S)
        roundmatrix(T)
        List_vectorS.append(S)
        List_vectorT.append(T)
    return List_vectorS,List_vectorT,Listbestx,Listbestf
    
            

#2.1 Case 1:  one state PSG + four states PSA]        

#Define function to optimize        
# define function to optimize
# case 1: s=1, t=4
def OptFunction1(x):      
   
    vectorS,vectorT=angle2ST(x,1,4)    
    
    W=np.array([
    [1+vectorS.item(1,0)*vectorT.item(1,0),vectorT.item(1,0)+vectorS.item(1,0),vectorS.item(2,0)*vectorT.item(2,0)+vectorS.item(3,0)*vectorT.item(3,0),vectorS.item(3,0)*vectorT.item(2,0)-vectorS.item(2,0)*vectorT.item(3,0) ],
    [1+vectorS.item(1,0)*vectorT.item(1,1),vectorT.item(1,1)+vectorS.item(1,0),vectorS.item(2,0)*vectorT.item(2,1)+vectorS.item(3,0)*vectorT.item(3,1),vectorS.item(3,0)*vectorT.item(2,1)-vectorS.item(2,0)*vectorT.item(3,1) ],
    [1+vectorS.item(1,0)*vectorT.item(1,2),vectorT.item(1,2)+vectorS.item(1,0),vectorS.item(2,0)*vectorT.item(2,2)+vectorS.item(3,0)*vectorT.item(3,2),vectorS.item(3,0)*vectorT.item(2,2)-vectorS.item(2,0)*vectorT.item(3,2) ],
    [1+vectorS.item(1,0)*vectorT.item(1,3),vectorT.item(1,3)+vectorS.item(1,0),vectorS.item(2,0)*vectorT.item(2,3)+vectorS.item(3,0)*vectorT.item(3,3),vectorS.item(3,0)*vectorT.item(2,3)-vectorS.item(2,0)*vectorT.item(3,3) ]
])
           
    W_inv=np.linalg.inv(W)
    Gauss=np.dot(W_inv,W_inv.T)
    
    f = np.trace(Gauss)
    
    return f            

List_vectorS_1,List_vectorT_1,List_bestx_1,List_bestf_1=Ellipsometry(OptFunction1,1,1,4)  



#2.1 Case 2 :  one state PSG + four states PSA]  

# define function to optimize
# case 2: s=2, t=2
def OptFunction2(x):      
   
    vectorS,vectorT=angle2ST(x,2,2)    
    
    W=np.array([
    [1+vectorS.item(1,0)*vectorT.item(1,0),vectorT.item(1,0)+vectorS.item(1,0),vectorS.item(2,0)*vectorT.item(2,0)+vectorS.item(3,0)*vectorT.item(3,0),vectorS.item(3,0)*vectorT.item(2,0)-vectorS.item(2,0)*vectorT.item(3,0) ],
    [1+vectorS.item(1,0)*vectorT.item(1,1),vectorT.item(1,1)+vectorS.item(1,0),vectorS.item(2,0)*vectorT.item(2,1)+vectorS.item(3,0)*vectorT.item(3,1),vectorS.item(3,0)*vectorT.item(2,1)-vectorS.item(2,0)*vectorT.item(3,1) ],
    [1+vectorS.item(1,1)*vectorT.item(1,0),vectorT.item(1,0)+vectorS.item(1,1),vectorS.item(2,1)*vectorT.item(2,0)+vectorS.item(3,1)*vectorT.item(3,0),vectorS.item(3,1)*vectorT.item(2,0)-vectorS.item(2,1)*vectorT.item(3,0) ],
    [1+vectorS.item(1,1)*vectorT.item(1,1),vectorT.item(1,1)+vectorS.item(1,1),vectorS.item(2,1)*vectorT.item(2,1)+vectorS.item(3,1)*vectorT.item(3,1),vectorS.item(3,1)*vectorT.item(2,1)-vectorS.item(2,1)*vectorT.item(3,1) ]
])
           
    W_inv=np.linalg.inv(W)
    Gauss=np.dot(W_inv,W_inv.T)
    
    f = np.trace(Gauss)
    
    return f          

# Case 2
List_vectorS_2,List_vectorT_2,List_bestx_2,List_bestf_2=Ellipsometry(OptFunction2,1,2,2)



