#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Librerias
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def pritnTabMax(Data,namesIMGE):
    n = Data.shape[0]

    for i in range(24):
        print('kodim' + str(i+1), ' & ', Data[i,0] , ' & ' ,int(Data[i,1]) , ' & ', Data[i,2] , ' \\\\' )
    #k=0    
    #for i in range(24,28):
    #    print(namesIMGE[k], ' & ', Data[i,0] , ' & ' ,Data[i,1] , ' & ', Data[i,2] , ' \\\\' )
    #    k=k+1
        


# In[3]:


Nprueb = '341B'
debn = 'debU'


# In[4]:


ruta  = './txt'+ Nprueb +'/'
namesIMGE = np.array(['butterfly.png','leaves.png', 'parrots.png','starfish.png'])
ls = []
lsgraf = []
resData = []


# In[5]:


k=0


# In[6]:


for i in range(1,25):#25
    
    nameimg = 'kodim' + str(i) + '_small.png'

    #nameimg =  namesIMGE[i]

    name = nameimg.replace('.png','.csv')

    fname = ruta + 'P'+ Nprueb + debn +'MAXval' + name
    maxPnsr = np.genfromtxt(fname, delimiter=',')
    
    #fname = ruta + 'P'+ Nprueb + 'denPNSR' + name 
    #plotY = np.genfromtxt(fname, delimiter=',')
    
    #Y = np.array(plotY).T
    
    #Tabla
    tabMax = np.array([ maxPnsr[:,1],maxPnsr[:,2], maxPnsr[:,3]])
    resData.append(tabMax)
    #print(tabMax)

LAMval = maxPnsr[:,0]

print (LAMval)
# In[7]:


for i in range(0):#4
    nameimg =  namesIMGE[i]

    name = nameimg.replace('.png','.csv')

    fname = ruta + 'P'+ Nprueb + debn +'MAXval' + name
    maxPnsr = np.genfromtxt(fname, delimiter=',')
    
    fname = ruta + 'P'+ Nprueb + debn +'PNSR' + name 
    plotY = np.genfromtxt(fname, delimiter=',')
    
    Y = np.array(plotY).T
    
    #Tabla
    tabMax = np.array([ maxPnsr[:,1],maxPnsr[:,2], Y[:,-1] ])
    resData.append(tabMax)
    #print(tabMax)


# In[8]:


tab = np.array(resData) 


# In[9]:


tab.shape


# In[10]:


total=tab.T


# In[11]:


total.shape


# In[12]:


total[0]


# In[13]:


#LAMval = [0,0.000001,0.00001,0.00003,0.00005,0.03,0.05]


# In[14]:


import pandas as pd 


# In[15]:


ruta = './outTXT/ExcelT_P'+ Nprueb + debn
for i in range(len(LAMval)):
    fname = ruta + str(LAMval[i])+ '.csv'
    tabla = total[i]
    fila_prom=np.mean(tabla,axis=1)
    tabla = np.concatenate((tabla.T,[fila_prom]),axis= 0)
    #np.savetxt(fname, tabla.T, delimiter =";",fmt='%0.3f; %i; %0.3f')
    TAB= pd.DataFrame(tabla)
    TAB[1] = TAB[1].astype('int64')
    TAB.to_csv(fname, sep=';', decimal=',',header=None,index=False,float_format='%.3f')


# In[16]:


TAB.dtypes


# In[17]:


TAB[1] = TAB[1].astype('int64')


# In[18]:


#pritnTabMax(tabla,namesIMGE)


# In[ ]:



