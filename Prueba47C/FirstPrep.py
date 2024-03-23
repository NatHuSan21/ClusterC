import os
    
carpNum = '47'
cluster = 'C'

#ruta = '/home/thomas.batard/SeminarioTesis2/Prueba' + carpNum + cluster
ruta = '/home/clirlab/natalia/Prueba' + carpNum + cluster

direcOutgraf = ruta + '/outGraf/UNIC/'

os.makedirs(direcOutgraf)

direcTXT = ruta + '/outTXT/'

os.mkdir(direcTXT)

direcRest = ruta + '/restoration/'

os.mkdir(direcRest)

for i in range(3):
    rutaRes = direcRest + 'P'+ carpNum + str(i) + cluster + '/'
    os.mkdir(rutaRes)
    rutaTxt = ruta + '/txt' + carpNum + str(i) + cluster + '/'
    os.mkdir(rutaTxt)
