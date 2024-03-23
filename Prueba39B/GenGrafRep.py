import cv2
import numpy as np
import matplotlib.pyplot as plt

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

Nprueb= '362B'
denb = 'debU'
num='0'

# In[6]:


ruta2 = './restoration/P'+ Nprueb +'/BestPnsr'+ denb +'_P'+ Nprueb +'_'
namesIMGE = np.array(['image_House256_greylevel.png', 'image_Lena512_greylevel.png', 'image_Peppers512_greylevel.png' , 'F16_GT_greylevel.png','image_Baboon512_greylevel.png'])
ls = []
lsgraf = []
resData = []

listaRep = range(1,5)
    
for k in listaRep:#25 #for k in range(1,25):#
    #nameimg = 'kodim18.png'#'image_Lena512rgb.png' #
    nameimg = 'kodim' + str(k) + '_small.png'
    
#for k in range(len(namesIMGE)):
#    nameimg =  namesIMGE[k]
    maxPnsr = []

    ruta  = './txt'+ Nprueb +'/'
    name = nameimg.replace('.png','.csv')

    fname = ruta + 'P' + Nprueb + denb + 'MAXval' + name
    #'/txt/1denMAXvalimage_Lena512rgb.csv'
    data = np.genfromtxt(fname, delimiter=',')
    maxPnsr.append(data)
    #print(data)

    pns = np.array(maxPnsr)

    valMX,q = (np.max(pns.T[1]), np.argmax(pns.T[1]))#(np.nanmax(pns.T[1]), np.nanargmax(pns.T[1]))
    #print(valMX,q)
    
    lamA = pns.T[0][q]

    lam = lamA[0]

    fila = pns.T[:,q]

    dif = np.max(pns.T[1]) - pns.T[1,0]
    
    fila = np.append(fila,dif)
    
    resData.append(fila)
    
    fname = ruta + 'P'+ Nprueb  + denb + 'PNSR' + name 
    plotY = np.genfromtxt(fname, delimiter=',')
    
    Y = np.array(plotY).T
    
    #Grafica
    x = np.arange(Y.shape[1])
    plt.plot(x,Y[q])
    plt.xlabel('Iteración')
    plt.ylabel('PSNR')
    plt.title(nameimg)
    namefig = './outGraf/UNIC/'+ str(k) +'lamb' + str(lam) + 'grafPNSR'+ Nprueb +'.png'
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

namefig = './outGraf/' + 'BestImgREP'+num +  Nprueb +'p1.png' 

printImgs(ls, 4, 2, namefig)


# In[9]:

namefig = './outGraf/' + 'BestImgREP'+ num + Nprueb +'p2.png' 

printImgs(ls[8:], 4, 2, namefig)


# In[10]:


namefig = './outGraf/' + 'BestImgREP'+num +  Nprueb +'p3.png' 

printImgs(ls[16:], 4, 2, namefig)


# In[11]:


namefig = './outGraf/' + 'BestImgREP'+ num + Nprueb +'E.png' 

#printImgs(ls[24:], 1, 5, namefig)


# In[12]:


namefig = './outGraf/' + 'grafPNSRImgREP'+ num + Nprueb +'p1.png' 

printGrafs(lsgraf, 4, 2, namefig)


# In[13]:


namefig = './outGraf/' + 'grafPNSRImgREP'+num +  Nprueb +'p2.png'

printGrafs(lsgraf[8:], 4, 2, namefig)


# In[14]:


namefig = './outGraf/' + 'grafPNSRImgREP'+num +  Nprueb +'p3.png' 

printGrafs(lsgraf[16:], 4, 2, namefig)


# In[15]:


namefig = './outGraf/' + 'grafPNSRImgREP'+ num + Nprueb +'E.png' 

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

pritnTabRep(resData,listaRep)
#pritnTab1(resData,24)


# In[21]:


ruta = './outTXT/ExcelResum'+ denb +  Nprueb 
fname = ruta + '.csv'
np.savetxt(fname, resData[:,:4], delimiter =",",fmt='%0.3f, %0.3f, %i, %0.3f')


# In[22]:


def pritnTab2(Data,namesIMGE,k):
    n = Data.shape[0]
    j = 0
    for i in range(k,n):
        print(namesIMGE[j], ' & ' ,Data[i,0] , ' & ' ,Data[i,1] , ' & ', Data[i,2] ,' & ', Data[i,3] ,' \\\\' )
        j+=1


# In[23]:


namesIMGE2 = np.array(['House256', 'Lena512', 'Peppers512' , 'F16\_GT','Baboon512'])

#pritnTab2(resData,namesIMGE2,24)


# In[24]:


freq=resData[:,0]

h=plt.hist(freq)

# In[29]:


(unique, counts) = np.unique(freq, return_counts=True)


# In[30]:


print(f'Únicos:     {unique}')
print(f'Frecuencia: {counts}')

etqCad = []
for val in unique:
    etqCad.append(str(val))
print(etqCad )

# In[31]:


parameters = {'xtick.labelsize': 25,
          'ytick.labelsize': 20}
plt.rcParams.update(parameters)

#fig, ax = plt.subplots()
plt.figure(figsize=(12,6))
etqs = etqCad #['0','0.01']

def autolabel(rects):
    """Funcion para agregar una etiqueta con el valor en cada barra"""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height-1),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',weight='bold', size=18)

#Añadimos las etiquetas para cada barra

grafb = plt.bar(etqs , counts)
#plt.legend()
autolabel(grafb)
plt.savefig('./outGraf/freqREP'+ Nprueb + num + '.png')

plt.show()


#Recortar imagenes


for i in range(1,4):
    namefig = './outGraf/' + 'grafPNSRImgREP'+ num +  Nprueb +'p'+str(i)+'.png' 

    image = cv2.imread(namefig)

    imageOut = image[235:1780,314:1530]

    #plt.axis('off')
    #plt.imshow(imageOut)
    #plt.savefig(namefig)
    #plt.imsave(namefig,  imageOut)
    cv2.imwrite(namefig,  imageOut)


