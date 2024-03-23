import torch
import torch.nn as nn
from math import sqrt
import torch.nn.init
from .common import *
from models import *


def derivative_forward_x_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_x_forward = u[1:,:]-u[:-1,:] #[H-1xW]
	u_x_forward = torch.cat((u_x_forward,torch.zeros(1,N).type(dtype)),axis=0) #[HxW]
	return u_x_forward

def derivative_backward_x_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_x_backward = u[1:,:]-u[:-1,:] #[H-1xW]
	u_x_backward = torch.cat((torch.zeros(1,N).type(dtype),u_x_backward),axis=0) #[HxW]
	return u_x_backward

def derivative_central_x_greylevel(u,dtype):

	return 0.5*(derivative_forward_x_greylevel(u,dtype) + derivative_backward_x_greylevel(u,dtype))

def derivative_forward_y_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_y_forward = u[:,1:]-u[:,:-1] #[HxW-1]
	u_y_forward = torch.cat((u_y_forward,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]
	return u_y_forward

def derivative_backward_y_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_y_backward = u[:,1:]-u[:,:-1] #[H-1xW]
	u_y_backward = torch.cat((torch.zeros(M,1).type(dtype),u_y_backward),axis=1) #[HxW]
	return u_y_backward

def derivative_central_y_greylevel(u,dtype):

	return 0.5*(derivative_forward_y_greylevel(u,dtype) + derivative_backward_y_greylevel(u,dtype))	

def derivative_xx_greylevel(u,dtype):

	return derivative_forward_x_greylevel(u,dtype) - derivative_backward_x_greylevel(u,dtype) 	

def derivative_yy_greylevel(u,dtype):

	return derivative_forward_y_greylevel(u,dtype) - derivative_backward_y_greylevel(u,dtype) 


def derivative_xy_greylevel(u,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_plus_plus = u[1:,1:] #[H-1xW-1]
	u_plus_plus = torch.cat((torch.zeros(1,N-1).type(dtype),u_plus_plus),axis=0) #[HxW-1]
	u_plus_plus = torch.cat((torch.zeros(M,1).type(dtype),u_plus_plus),axis=1)  #[HxW]

	u_minus_minus = u[:-1,:-1] #[H-1xW-1]
	u_minus_minus = torch.cat((u_minus_minus,torch.zeros(1,N-1).type(dtype)),axis=0) #[HxW-1]
	u_minus_minus = torch.cat((u_minus_minus,torch.zeros(M,1).type(dtype)),axis=1)  #[HxW]

	u_plus_minus = u[1:,:-1] #[H-1xW-1]
	u_plus_minus = torch.cat((torch.zeros(1,N-1).type(dtype),u_plus_minus),axis=0) #[HxW-1]
	u_plus_minus = torch.cat((u_plus_minus, torch.zeros(M,1).type(dtype)),axis=1) #[HxW]

	u_minus_plus = u[:-1,1:] #[H-1xW-1]
	u_minus_plus = torch.cat((u_minus_plus,torch.zeros(1,N-1).type(dtype),),axis=0) #[HxW-1]
	u_minus_plus = torch.cat((torch.zeros(M,1).type(dtype),u_minus_plus),axis=1) #[HxW]
														
	u_xy = (u_plus_plus+u_minus_minus-u_plus_minus-u_minus_plus)/4.	
	return u_xy


def riemannianmetric_greylevel(u,beta,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_x = derivative_central_x_greylevel(u,dtype)
	u_y = derivative_central_y_greylevel(u,dtype)

	g11 = torch.ones((M,N)).type(dtype) + beta*torch.square(u_x)
	g12 = beta*torch.mul(u_x,u_y)
	g22 = torch.ones((M,N)).type(dtype) + beta*torch.square(u_y)

	return g11,g12,g22

def riemanniangradient_greylevel(u,g11,g12,g22,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	detg = torch.mul(g11,g22) - torch.square(g12)

	invdetg = torch.div(torch.ones((M,N)).type(dtype),detg)
	invg11 = torch.mul(invdetg,g22)
	invg12 = - torch.mul(invdetg,g12)
	invg22 = torch.mul(invdetg,g11)

	u_x_forward = u[1:,:]-u[:-1,:] #[H-1xW]
	u_x_forward = torch.cat((u_x_forward,torch.zeros(1,N).type(dtype)),axis=0) #[HxW]

	u_y_forward = u[:,1:]-u[:,:-1] #[HxW-1]
	u_y_forward = torch.cat((u_y_forward,torch.zeros(M,1).type(dtype)),axis=1) #[HxW]

	return torch.mul(invg11,u_x_forward) + torch.mul(invg12,u_y_forward), torch.mul(invg12,u_x_forward) + torch.mul(invg22,u_y_forward)



def riemanniandivergence_greylevel(U1,U2,g11,g12,g22,dtype):

	M = U1.size(dim=0)
	N = U2.size(dim=1)

	sqrt_abs_detg = torch.sqrt(torch.abs(torch.mul(g11,g22) - torch.square(g12)))
	component1 = torch.mul(sqrt_abs_detg,U1)
	component2 = torch.mul(sqrt_abs_detg,U2)

	component1_x_backward = component1[1:,:]-component1[:-1,:]
	component1_x_backward = torch.cat((torch.zeros(1,N).type(dtype),component1_x_backward),axis=0)

	component2_y_backward = component2[:,1:]-component2[:,:-1]
	component2_y_backward = torch.cat((torch.zeros(M,1).type(dtype),component2_y_backward),axis=1)

	return torch.mul(torch.div(torch.ones(M,N).type(dtype),sqrt_abs_detg),component1_x_backward+component2_y_backward)


def orthogonal_moving_frame(g11,g12,g22,dtype):

	M = g11.size(dim=0)
	N = g11.size(dim=1)

	detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[1x1xHxW]

	lambda_plus = (1./2.)*(g11+g22+torch.sqrt(0.00001*torch.ones((M,N)).type(dtype)+torch.square(g11+g22)-4.*detg))
	lambda_moins = (1./2.)*(g11+g22-torch.sqrt(0.00001*torch.ones((M,N)).type(dtype)+torch.square(g11+g22)-4.*detg))

	a = torch.div(lambda_plus-g22,0.00001*torch.ones((M,N)).type(dtype)+torch.mul(torch.sqrt(lambda_plus),torch.sqrt(torch.square(g12)+torch.square(lambda_plus-g22))))
	b = torch.div(g12,0.00001*torch.ones((M,N)).type(dtype)+torch.mul(torch.sqrt(lambda_plus),torch.sqrt(torch.square(g12)+torch.square(lambda_plus-g22))))
	c = torch.mul(torch.sign(g12), 0.00001*torch.ones((M,N)).type(dtype)+torch.div(lambda_moins-g22,torch.mul(torch.sqrt(lambda_moins),torch.sqrt(torch.square(g12)+torch.square(lambda_moins-g22)))))
	d = torch.mul(torch.sign(g12), 0.00001*torch.ones((M,N)).type(dtype)+torch.div(g12,torch.mul(torch.sqrt(lambda_moins),torch.sqrt(torch.square(g12)+torch.square(lambda_moins-g22)))))

 
	'''print(torch.max(lambda_plus.detach()))
	print(torch.max(lambda_moins.detach()))
	print(torch.min(lambda_plus.detach()))
	print(torch.min(lambda_moins.detach()))

	print(torch.max( (torch.square(a)+torch.square(b)).detach()))
	print(torch.min( (torch.square(a)+torch.square(b)).detach()))
	print(torch.max( (torch.square(c)+torch.square(d)).detach()))
	print(torch.min( (torch.square(c)+torch.square(d)).detach() ))
	print(torch.max( (torch.mul(a,c)+torch.mul(b,d)).detach() ))
	print(torch.min( (torch.mul(a,c)+torch.mul(b,d)).detach() ))'''


	return a,b,c,d



def laplacebeltrami_greylevel(u,g11,g12,g22,dtype):

	U1,U2 = riemanniangradient_greylevel(u,g11,g12,g22,dtype)

	return riemanniandivergence_greylevel(U1,U2,g11,g12,g22,dtype)


def gaussiancurvature_greylevel(u,beta,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)
	
	u_x = derivative_central_x_greylevel(u)
	u_y = derivative_central_y_greylevel(u)

	u_xx = derivative_xx_greylevel(u)
	u_xy = derivative_xy_greylevel(u)
	u_yy = derivative_yy_greylevel(u)

	num = torch.mul(u_xx,u_yy)-torch.square(u_xy)
	den = (beta**(3/2))*torch.square((1./beta)*torch.ones(M,N).type(dtype) + torch.square(u_x) + torch.square(u_y))
																		 
	return torch.div(num,den)


def meancurvature_greylevel(u,beta,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	u_x = derivative_central_x_greylevel(u)
	u_y = derivative_central_y_greylevel(u)

	u_xx = derivative_xx_greylevel(u)
	u_xy = derivative_xy_greylevel(u)
	u_yy = derivative_yy_greylevel(u)

	num = torch.mul(u_xx,torch.ones(M,N).type(dtype)+beta*(torch.square(u_y))) \
		-2.*(beta)*torch.mul(torch.mul(u_xy,u_x),u_y) \
		+ torch.mul(u_yy,torch.ones(M,N).type(dtype)+beta*torch.square(u_x))

	den = 2.*(beta**(2/3))*torch.pow((1./(beta))*torch.ones(M,N).type(dtype)+torch.square(u_x)+torch.square(u_y), 3.2)


	return torch.div(num,den)



def Levi_Cevita_connection_coefficients_standard_frame(u,beta,dtype):

	M = u.size(dim=0)
	N = u.size(dim=1)

	
	g11,g12,g22 = riemannianmetric_greylevel(u,beta,dtype)

	detg = torch.mul(g11,g22) - torch.mul(g12,g12) 
	invdetg = torch.div(torch.ones(M,N).type(dtype),detg)

	u_x = derivative_central_x_greylevel(u,dtype)
	u_y = derivative_central_y_greylevel(u,dtype)
	u_xx = derivative_xx_greylevel(u,dtype)
	u_xy = derivative_xy_greylevel(u,dtype)
	u_yy = derivative_yy_greylevel(u,dtype)


	Gamma1_11 = beta*torch.mul(invdetg,torch.mul(u_x,u_xx)) # [1x1xHxW]
	Gamma2_22 = beta*torch.mul(invdetg,torch.mul(u_y,u_yy)) # [1x1xHxW]
	Gamma1_12 = beta*torch.mul(invdetg,torch.mul(u_x,u_xy)) # [1x1xHxW]
	Gamma1_21 = Gamma1_12 # [1x1xHxW]
	Gamma2_12 = beta*torch.mul(invdetg,torch.mul(u_y,u_xy)) # [1x1xHxW]
	Gamma2_21 = Gamma2_12 # [1x1xHxW]
	Gamma1_22 = beta*torch.mul(invdetg,torch.mul(u_x,u_yy)) # [1x1xHxW]
	Gamma2_11 = beta*torch.mul(invdetg,torch.mul(u_y,u_xx)) # [1x1xHxW]

	return Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11


def normriemannianhessianmatrix_greylevel(u,beta,dtype):

	Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11 = Levi_Cevita_connection_coefficients_standard_frame(u,beta,dtype)

	M = u.size(dim=0)
	N = u.size(dim=1)

	g11,g12,g22 = riemannianmetric_greylevel(u,beta,dtype)

	detg = torch.mul(g11,g22) - torch.mul(g12,g12) 
	invdetg = torch.div(torch.ones(M,N).type(dtype),detg)
	invg11 = torch.mul(invdetg,g22)
	invg12 = - torch.mul(invdetg,g12)
	invg22 = torch.mul(invdetg,g11)
	
	u_x = derivative_central_x_greylevel(u,dtype)
	u_y = derivative_central_y_greylevel(u,dtype)

	u_xx = derivative_xx_greylevel(u,dtype)
	u_xy = derivative_xy_greylevel(u,dtype)
	u_yy = derivative_yy_greylevel(u,dtype)

	Hess_u_11 = u_xx - torch.mul(Gamma1_11,u_x)-torch.mul(Gamma2_11,u_y)
	Hess_u_12 = u_xy - torch.mul(Gamma1_12,u_x)-torch.mul(Gamma2_12,u_y)
	Hess_u_22 = u_yy - torch.mul(Gamma1_22,u_x)-torch.mul(Gamma2_22,u_y)

	norm_Hess_u_squared = torch.mul(torch.square(invg11),torch.square(Hess_u_11)) + 4.*torch.mul(invg11,torch.mul(invg12,torch.mul(Hess_u_11,Hess_u_12))) \
	+ 4.*torch.mul(invg12,torch.mul(invg12,torch.mul(Hess_u_12,Hess_u_12))) + 4.*torch.mul(invg12,torch.mul(invg22,torch.mul(Hess_u_12,Hess_u_22))) \
	+ 2.*torch.mul(invg11,torch.mul(invg22,torch.mul(Hess_u_11,Hess_u_22))) + torch.mul(torch.square(invg22),torch.square(Hess_u_22)) 

	norm_Hess_u_reg = torch.sqrt(0.0000001*torch.ones(M,N).type(dtype)+norm_Hess_u_squared)

	return norm_Hess_u_reg








