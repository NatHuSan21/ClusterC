import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def printImgs(images, n=1, m=1, title=''):
    
    images = images[:n*m]
    fig, axs = plt.subplots(
        nrows=n, ncols=m, 
        figsize=(15, 20), 
        subplot_kw={
            'xticks': [], 
            'yticks': []
        }
    )
    for ax, image in zip(axs.flat, images):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.savefig(title)    
    plt.show()


def printGrafs(images, n=1, m=1, title=''):
    
    images = images[:n*m]
    fig, axs = plt.subplots(
        nrows=n, ncols=m, 
        figsize=(18, 20), 
        subplot_kw={
            'xticks': [], 
            'yticks': []
        }
    )
    for ax, image in zip(axs.flat, images):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.savefig(title)    
    plt.show()


# In[5]:

Nprueb = '421C'
denb = 'den25'#'den15'
lim = 25
ValLamb = 0.03

num_iter = 8000


listaRep = [6,15]

ruta2 = './restoration/P'+ Nprueb +'/BestPnsr'+ denb+'_P'+ Nprueb +'_'
#ruta2 = './restoration/P'+ Nprueb +'/BestPnsr_P'+ Nprueb +'_'

namesIMGE = np.array(['image_House256_greylevel.png', 'image_Lena512_greylevel.png', 'image_Peppers512_greylevel.png' , 'F16_GT_greylevel.png','image_Baboon512_greylevel.png'])


ls = []
lsgraf = []
resData = []
lsYgraf = []

k=1
if denb[:3] == 'den':
    nameimg = 'kodim' + str(k) + '.png'
else:
    nameimg = 'kodim' + str(k) + '_small.png'
ruta  = './txt'+ Nprueb +'/'
name = nameimg.replace('.png','.csv')
#fname = ruta + 'P' + Nprueb + 'denMAXval' + name
fname = ruta + 'P' + Nprueb + denb +'MAXval' + name
#'/txt/1denMAXvalimage_Lena512rgb.csv'
data = np.genfromtxt(fname, delimiter=',')
    #print(data)

pns = np.array(data)

tabPrueb = pns
#print(tabPrueb)
print(tabPrueb.shape)
print(tabPrueb[:,0])
LamVals = tabPrueb[:,0]
posLam = np.where(LamVals == ValLamb)
print(posLam)
q=posLam[0][0]



for k in range(1,lim):#for k in listaRep:#25
    #nameimg = 'kodim18.png'#'image_Lena512rgb.png' #
    if denb[:3] == 'den':
        nameimg = 'kodim' + str(k) + '.png'
    else:
        nameimg = 'kodim' + str(k) + '_small.png'
#for k in range(len(namesIMGE)):
#    nameimg =  namesIMGE[k]
    maxPnsr = []

    ruta  = './txt'+ Nprueb +'/'
    if k in listaRep:
        name = nameimg.replace('.png','_2.csv')
    else:
        name = nameimg.replace('.png','.csv')
    #fname = ruta + 'P' + Nprueb + 'denMAXval' + name
    fname = ruta + 'P' + Nprueb + denb +'MAXval' + name
    #'/txt/1denMAXvalimage_Lena512rgb.csv'
    data = np.genfromtxt(fname, delimiter=',')
    maxPnsr.append(data)
    #print(data)

    pns = np.array(maxPnsr)

    lam = ValLamb

    print(pns.shape)

    if k in listaRep:
        fila = pns.T[:]
    else:
        fila = pns.T[:,q]

    dif = 0.0#pns.T[1,q] - pns.T[1,0]
    
    fila = np.append(fila,dif)
    
    resData.append(fila)
    
    fname = ruta + 'P'+ Nprueb  + denb +'PSNR' + name 
    plotY = np.genfromtxt(fname, delimiter=',')
    
    Y = np.array(plotY).T

    if k in listaRep:
        Ygraf = Y
    else:
        Ygraf = Y[q]

    lsYgraf.append(Ygraf)

    #print(Y[q])
    #Grafica
    x = np.arange(num_iter)#(30000)#
    plt.plot(x, Ygraf, "g",linewidth=2)#Y)#
    plt.xlabel('Iteración')
    plt.ylabel('PSNR')
    plt.title('Lam' + str(ValLamb) + ' '+ nameimg)
    namefig = './outGraf/UNIC/'+ str(k)+ 'mxLamb' + str(lam) + 'grafPNSR'+ Nprueb +'.png'
    plt.savefig(namefig)
    #plt.show()
    plt.clf()

    #Imagen grafica
    imgOut = cv2.imread(namefig)
    lsgraf.append(imgOut)
    
    #Imagen
    if lam !=0.0:
        fname = ruta2 + str(lam)  + nameimg
    else:
        fname = ruta2 + '0'  + nameimg
    #print(fname)
    imgOut = cv2.imread(fname)
    
    ls.append(imgOut)

 
# In[8]:

###Generar imagenes de graficas en conjunto

namefig = './outGraf/' + 'mxLamb' + 'BestImg'+ Nprueb +'p1.png' 

printImgs(ls, 4, 2, namefig)


# In[9]:

namefig = './outGraf/' + 'mxLamb' + 'BestImg'+ Nprueb +'p2.png' 

printImgs(ls[8:], 4, 2, namefig)


# In[10]:


namefig = './outGraf/' + 'mxLamb' + 'BestImg'+ Nprueb +'p3.png' 

printImgs(ls[16:], 4, 2, namefig)


# In[11]:


namefig = './outGraf/' + 'mxLamb' + 'BestImg'+ Nprueb +'E.png' 

#printImgs(ls[24:], 1, 5, namefig)


# In[12]:


namefig = './outGraf/' + 'mxLamb' + 'grafPNSRImg'+ Nprueb +'p1.png' 

printGrafs(lsgraf, 4, 2, namefig)


# In[13]:


namefig = './outGraf/' + 'mxLamb' + 'grafPNSRImg'+ Nprueb +'p2.png'

printGrafs(lsgraf[8:], 4, 2, namefig)


# In[14]:


namefig = './outGraf/' + 'mxLamb' + 'grafPNSRImg'+ Nprueb +'p3.png' 

printGrafs(lsgraf[16:], 4, 2, namefig)


# In[15]:


namefig = './outGraf/' + 'mxLamb' + 'grafPNSRImg'+ Nprueb +'E.png' 

#printGrafs(lsgraf[24:], 1, 5, namefig)


# In[16]:
#Tabla de datos

resData=np.array(resData)


# In[19]:


def pritnTab1(Data,k):
    #n = Data.shape[0]
    for i in range(k):
        print('kodim' + str(i+1), ' & ' ,Data[i,0] , ' & ' ,Data[i,1] , ' & ', int(Data[i,2]) ,' & ', Data[i,3] ,' \\\\' )

def pritnTabRep(Data,listaRep):
    #n = Data.shape[0]
    k=len(listaRep)
    for i in range(k):
        print('kodim' + str(listaRep[i]), ' & ' ,Data[i,0] , ' & ' ,Data[i,1] , ' & ', int(Data[i,2]) ,' & ', Data[i,3] ,' \\\\' )



# In[20]:

#pritnTabRep(resData,listaRep)
pritnTab1(resData,lim-1)


# In[21]:

print(resData.shape)
ruta = './outTXT/ExcelResumBest'+ 'mxLamb' + str(ValLamb) + '_P'+ Nprueb + denb
fname = ruta + '.csv'
#np.savetxt(fname, resData[:,:4], delimiter =",",fmt='%0.3f, %0.3f, %i, %0.3f')
fila_prom = np.mean(resData,axis=0)
print(fila_prom)
tabla = np.concatenate((resData,[fila_prom]),axis= 0)
TAB= pd.DataFrame(tabla[:,:4])
TAB[0] = TAB[0].astype('str')
TAB[2] = TAB[2].astype('int64')
TAB.to_csv(fname, sep=';', decimal=',',header=None,index=False,float_format='%.3f')
#np.savetxt(fname, resData[:,:4], delimiter =",",fmt='%0.3f, %0.3f, %i, %0.3f')

fname = ruta + 'SF.csv'
np.savetxt(fname, tabla[:,:4], delimiter =",",fmt ='% s')

GraficasY = np.array(lsYgraf)
####Guardar datos de Graficas
ruta = './outTXT/DataGraf'+ 'mxLamb' + str(ValLamb) +'_P'+ Nprueb + denb
fname = ruta + '.csv'
np.savetxt(fname, GraficasY.T, delimiter =",",fmt ='% s')


#Recortar imagenes

def RecorImg(rutaNameImg):

    for i in range(1,4):
        namefig = rutaNameImg + str(i) + '.png' 

        image = cv2.imread(namefig)

        imageOut = image[235:1780,314:1530]

        #plt.axis('off')
        #plt.imshow(imageOut)
        #plt.savefig(namefig)
        #plt.imsave(namefig,  imageOut)
        cv2.imwrite(namefig,  imageOut)

#rutaNameImg = './outGraf/' + 'grafPNSRImg'+ Nprueb +'p'
#RecorImg(rutaNameImg)

rutaNameImg = './outGraf/' + 'mxLamb' + 'grafPNSRImg'+ Nprueb +'p'
RecorImg(rutaNameImg)




