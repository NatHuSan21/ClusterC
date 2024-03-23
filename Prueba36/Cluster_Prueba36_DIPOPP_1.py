from __future__ import print_function

import os
import cv2
import numpy as np
from models import *
from models.dip_reg import*

import torch
import torch.optim

import kornia

from utils.denoising_utils import *
from utils.blur_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


Nprueb= '361'

denb = 'debU'

##TAMBIEN CAMBIAR ARCHIVOS csv
#namesIMGE = np.array(['image_House256_greylevel.png', 'image_Lena512_greylevel.png', 'image_Peppers512_greylevel.png' , 'F16_GT_greylevel.png','image_Baboon512_greylevel.png'])
#for k in range(len(namesIMGE)):
#    nameimg =  namesIMGE[k]

    
listaRep = [6]#range(1,15)#
    
for k in listaRep:#for k in range(1,25):#25
    #nameimg = 'kodim18.png'#'image_Lena512rgb.png' #
    nameimg = 'kodim' + str(k) + '_small.png'

	#Lectura de la imagen clean y degradada
	#ruta = '/content/gdrive/MyDrive/SeminariodetesisI/Codigo/DIP_VBTV/data/deblurring/' 
    ruta = './data/data_deblurring_thomas/deblurring/'
    fname = ruta + nameimg
    img_pil = crop_image(get_image(fname,-1)[0], d=32)
    img_width,img_height = img_pil.size
    
    img_np = pil_to_np(img_pil)


    #ruta = './data/noisy_25/' 
    ruta = './data/data_deblurring_thomas/deblurring/' 
    name = nameimg.replace('.png','_blurred_uniform.png')
    fname = ruta + name
    img_degraded_pil = crop_image(get_image(fname,-1)[0], d=32)
    img_degraded_np = pil_to_np(img_degraded_pil)
    img_degraded_torch = np_to_torch(img_degraded_np).type(dtype)

    # Generta the degradation operator H
    NOISE_SIGMA = 2**.5
    BLUR_TYPE = 'uniform_blur'#'gauss_blur'
    GRAY_SCALE = False 
    USE_FOURIER = False

    ORIGINAL  = 'Clean'
    CORRUPTED = 'Blurred'

    data_dict = { ORIGINAL: Data(img_np), 
                        CORRUPTED: Data(img_degraded_np, compare_PSNR(img_np, img_degraded_np, on_y=(not GRAY_SCALE), gray_scale=GRAY_SCALE)) }

    H = get_h(data_dict[CORRUPTED].img.shape[0], BLUR_TYPE, USE_FOURIER, dtype)

    # Conversion from rgb to opponent space
    img_degraded_np2 = np.zeros((img_degraded_np.shape))
    img_degraded_np2[0,:,:]=(1./(0.6*sqrt(3)))*(np.sum(img_degraded_np,axis=0))
    img_degraded_np2[1,:,:]=(1/np.sqrt(2))*(img_degraded_np[0,:,:]-img_degraded_np[1,:,:])
    img_degraded_np2[2,:,:]=(1/np.sqrt(6))*(img_degraded_np[0,:,:]+img_degraded_np[1,:,:]-2*img_degraded_np[2,:,:])
    img_degraded_torch2 = np_to_torch(img_degraded_np2).type(dtype)

    #Arreglos de datos
    plotY = []
    maxPnsr = []

    #Valores Lambda de prueba
        
    LAMval = [0]#,0.000001,0.00001,0.00003,0.00005,0.0001,0.00025,0.0005,0.001,0.005,0.0025, 0.03,0.05]

    for lam in LAMval:
        # Setup
        INPUT = 'noise'
        pad = 'reflection'
        OPT_OVER ='net'

        reg_noise_std = 0.01
        LR = 0.001

        Lambda = lam

        OPTIMIZER = 'adam' 
        exp_weight = 0.99

        num_iter = 30000
        input_depth = 32


        full_net =  UnknownOperatorV4OPP(input_depth, dtype, pad, height=img_height, width= img_width, upsample_mode='bilinear').type(dtype)
        net_input = get_noise(input_depth, INPUT, (img_height,img_width)).type(dtype).detach()

        # Compute number of parameters
        s  = sum([np.prod(list(p.size())) for p in full_net.parameters()]); 

        # Loss
        mse = torch.nn.MSELoss().type(dtype)
        mae = torch.nn.L1Loss().type(dtype)

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        out_img_avg = img_degraded_np

        pnsrLS = []

        valorMax = 0
        imgPNSRmax_np = np.zeros(out_img_avg.shape)

        i = 0
        def closure():
                
            global i, exp_weight, out_img_avg, net_input

            #Para guardar mejor imagen
            global valorMax, imgPNSRmax_np
                
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)    

            net_output, regularizer = full_net(net_input)	

            loss_regularizer = mae(regularizer,torch.zeros(1,img_height,img_width).type(dtype))
            
            loss_dataterm = mse(H(net_output),img_degraded_torch)
            #loss_dataterm = mse(net_output,img_degraded_torch)

            # Conversion of the network output from rgb to opp
            opp1=(1./(0.6*sqrt(3)))*torch.unsqueeze(torch.sum(net_output,dim=1),dim=1)
            opp2= (1/sqrt(2))*torch.narrow(net_output,1,0,1) - (1/sqrt(2))*torch.narrow(net_output,1,1,1)
            opp3= (1/sqrt(6))*torch.narrow(net_output,1,0,1) + (1/sqrt(6))*torch.narrow(net_output,1,1,1) - (2/sqrt(6))*torch.narrow(net_output,1,2,1)
            opp=torch.cat((opp1,opp2,opp3),dim=1)

            loss_dataterm = mse(H(opp),img_degraded_torch2)

            total_loss = loss_dataterm + Lambda*loss_regularizer
            total_loss.backward(retain_graph=True)
						
            out_img_avg = out_img_avg * exp_weight + net_output.detach().cpu().numpy()[0] * (1 - exp_weight)
			
            #print(img_np.shape)
            #print(out_img_avg.shape)
            val = compare_psnr(img_np,out_img_avg)#compare_pnsr
            pnsrLS.append(val)
                
            if valorMax < val : #Guardar imagen con mayor pnsr
                imgPNSRmax_np = out_img_avg
                valorMax = val


            i += 1

            return total_loss

        p = get_params(OPT_OVER, full_net, net_input, input_depth)
        optimize(OPTIMIZER, p, closure, LR, num_iter)

        #Save image output
        out_img_avg_pil = np_to_pil(out_img_avg)
        #ruta = './restoration/P290/denoised_P290_'
        ruta = './restoration/P'+ Nprueb +'/'+ denb +'_P'+ Nprueb +'_'
        fname = ruta + str(Lambda)  + nameimg 
        out_img_avg_pil.save(fname)

        #Save best image 
        imgPNSRmax_pil = np_to_pil(imgPNSRmax_np)
        #fname = './restoration/P290/BestPnsr_P290_' + str(Lambda) + nameimg
        fname = './restoration/P'+ Nprueb +'/BestPnsr'+ denb +'_P'+ Nprueb +'_' + str(Lambda) + nameimg
        imgPNSRmax_pil.save(fname)

        #Datos para gráfica
        y = pnsrLS
        plotY.append(y)
        
        #Datos para tabla
        maxIt = np.array([Lambda,np.max(y),np.argmax(y),y[-1]])
        maxPnsr.append(maxIt)
 
    #Guardar datos de gráfica
    Y = np.array(plotY)
    #ruta = './txt290/'
    ruta = './txt'+ Nprueb +'/'
    name = nameimg.replace('.png','.csv')
    #fname = ruta + 'P290denPNSR' + name
    fname = ruta + 'P'+ Nprueb + denb +'PNSR' + name
    np.savetxt(fname, Y.T, delimiter =",",fmt ='% s')
            
    #Guardar datos para tabla
    MX = np.array(maxPnsr)
    #ruta = './txt290/'
    ruta = './txt'+ Nprueb +'/'
    name = nameimg.replace('.png','.csv')
    #fname = ruta  + 'P290denMAXval' + name
    fname = ruta  + 'P'+ Nprueb + denb +'MAXval' + name
    np.savetxt(fname, MX, delimiter =",",fmt ='% s')

