import torch
import torch.nn as nn
from math import sqrt
import torch.nn.init
import kornia
from .common import *
from models import *
from models.regularizers import *


class DIP(nn.Module):

	def __init__(self,input_depth, pad, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(DIP,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.input_depth=input_depth

	def forward(self, input):
		
		return self.net(input)


class VectorialTotalVariationSobel(nn.Module):	

	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(VectorialTotalVariationSobel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth=input_depth
		self.dtype=dtype

	def forward(self, input):

		output = self.net(input)

		differential = kornia.filters.SpatialGradient()(output)
		differential_squared = torch.mul(differential,differential)

		norm_squared = torch.sum(torch.sum(differential_squared,dim=1),dim=1)
		norm_squared_regularized = norm_squared+0.00001*torch.ones((1,self.height,self.width)).type(self.dtype)
		norm = torch.sqrt(norm_squared_regularized)
		return output, norm


class RiemannianTotalVariationGreyLevelSobel(nn.Module):
	def __init__(self,input_depth, dtype, pad, beta, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianTotalVariationGreyLevelSobel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]

		differential_output = kornia.filters.SpatialGradient()(output) #[1x1x2xHxW]
				
		output_x = torch.narrow(differential_output,2,0,1) #[1x1x1xHxW]	
		output_x = torch.squeeze(output_x) #[HxW]
		output_y = torch.narrow(differential_output,2,1,1) #[1x1x1xHxW]
		output_y = torch.squeeze(output_y) #[HxW]

		output_x_squared = torch.square(output_x)	#[HxW]
		output_y_squared = torch.square(output_y)	#[HxW]	

		g11 = torch.ones((self.height,self.width)).type(self.dtype) + self.beta*output_x_squared
		g22 = torch.ones((self.height,self.width)).type(self.dtype) + self.beta*output_y_squared
		g12 = self.beta*torch.mul(output_x,output_y)
		detg = torch.mul(g11,g22) - torch.mul(g12,g12)

		invdetg = torch.div(torch.ones((self.height,self.width)).type(self.dtype),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)

		norm_squared = torch.mul(invg11,output_x_squared) \
		                   +2*torch.mul(invg12,torch.mul(output_x,output_y)) \
		                   +torch.mul(invg22,output_y_squared)

		norm_squared_regularized = norm_squared+0.00001*torch.ones((self.height,self.width)).type(self.dtype)
		norm = torch.sqrt(norm_squared_regularized)
		norm = torch.unsqueeze(norm,dim=0)

		return output, norm


class TotalVariationGreyLevelCentralDerivatives(nn.Module):
	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(TotalVariationGreyLevelCentralDerivatives,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]

		output2 = torch.squeeze(output) #[HxW]

		output2_x = derivative_central_x_greylevel(output2,self.dtype) #[HxW]
		output2_y = derivative_central_y_greylevel(output2,self.dtype) #[HxW]

		tv = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype)+ torch.square(output2_x) + torch.square(output2_y)) #[HxW]
		tv = torch.unsqueeze(tv,dim=0) #[1xHxW]

		return output, tv


class InvariantGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, luminance, differential_luminance_x,differential_luminance_y, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(InvariantGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype
		self.luminance0 = luminance
		self.differential_luminance0_x = differential_luminance_x 
		self.differential_luminance0_y = differential_luminance_y 

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]

		output2 = torch.squeeze(output) #[HxW]

		output2_x = derivative_central_x_greylevel(output2,self.dtype) #[HxW]
		output2_y = derivative_central_y_greylevel(output2,self.dtype) #[HxW]

		omega_x = - torch.div(torch.mul(self.luminance0, torch.mul(output2_x,self.differential_luminance0_x)),0.001*torch.ones((self.height,self.width)).type(self.dtype) + torch.mul(output2,torch.square(self.differential_luminance0_x)+torch.square(self.differential_luminance0_y)))
		omega_y = - torch.div(torch.mul(self.luminance0, torch.mul(output2_y,self.differential_luminance0_y)),0.001*torch.ones((self.height,self.width)).type(self.dtype) + torch.mul(output2,torch.square(self.differential_luminance0_x)+torch.square(self.differential_luminance0_y)))

		covariant_derivative_x = output2_x + torch.mul(omega_x,output2)
		covariant_derivative_y = output2_y + torch.mul(omega_y,output2)


		norm = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype)+ torch.square(covariant_derivative_x) + torch.square(covariant_derivative_y)) #[HxW]
		norm2 = torch.unsqueeze(norm,dim=0) #[1xHxW]

		return output, norm2



class RiemannianTotalVariationGreyLevelCentralDerivatives(nn.Module):
	def __init__(self,input_depth, dtype, pad, beta, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianTotalVariationGreyLevelCentralDerivatives,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]

		output2 = torch.squeeze(output) #[HxW]

		output2_x = derivative_central_x_greylevel(output2,self.dtype)
		output2_y = derivative_central_y_greylevel(output2,self.dtype)
		g11,g12,g22 = riemannianmetric_greylevel(output2,self.beta,self.dtype)

		detg = torch.mul(g11,g22) - torch.square(g12)

		invdetg = torch.div(torch.ones((self.height,self.width)).type(self.dtype),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)

		norm_squared = torch.mul(invg11,torch.square(output2_x)) \
		                   +2*torch.mul(invg12,torch.mul(output2_x,output2_y)) \
		                   +torch.mul(invg22,torch.square(output2_y))

		norm_squared_regularized = norm_squared+0.00001*torch.ones((self.height,self.width)).type(self.dtype)
		norm = torch.sqrt(norm_squared_regularized)
		norm = torch.unsqueeze(norm,dim=0)

		return output, norm


class LaplaceBeltramiGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, beta, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(LaplaceBeltramiGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)


		self.input_depth = input_depth
		self.beta = beta
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]
		output2 = torch.squeeze(output) #[HxW]

		g11,g12,g22 = riemannianmetric_greylevel(output2,self.beta,self.dtype)
		Delta_output2 = laplacebeltrami_greylevel(output2,g11,g12,g22,self.dtype)
		regularizer = torch.unsqueeze(torch.abs(Delta_output2),dim=0)

		return output, regularizer


class RiemannianHessianGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, beta, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianHessianGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.input_depth = input_depth
		self.beta = beta
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]
		output2 = torch.squeeze(output) #[HxW]
		##########checar aqui no encuentra la funcion	
		#norm_hessian_reg_output2 = normriemannianhessianmatrix_greylevel(output2,self.beta,self.dtype)
		norm_hessian_reg_output2 =output2
		regularizer = torch.unsqueeze(norm_hessian_reg_output2,dim=0)

		return output, regularizer


class RiemannianGradientHessianGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, beta, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianGradientHessianGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.beta = beta
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]
		output2 = torch.squeeze(output) #[HxW]

		output2_x = derivative_central_x_greylevel(output2,self.dtype)
		output2_y = derivative_central_y_greylevel(output2,self.dtype)
		g11,g12,g22 = riemannianmetric_greylevel(output2,self.beta,self.dtype)

		detg = torch.mul(g11,g22) - torch.square(g12)

		invdetg = torch.div(torch.ones((self.height,self.width)).type(self.dtype),detg)
		invg11 = torch.mul(invdetg,g22)
		invg12 = - torch.mul(invdetg,g12)
		invg22 = torch.mul(invdetg,g11)

		norm_squared = torch.mul(invg11,torch.square(output2_x)) \
		                   +2*torch.mul(invg12,torch.mul(output2_x,output2_y)) \
		                   +torch.mul(invg22,torch.square(output2_y))

		norm_squared_regularized = norm_squared+0.00001*torch.ones((self.height,self.width)).type(self.dtype)
		norm = torch.sqrt(norm_squared_regularized)
		regularizer1 = torch.unsqueeze(norm,dim=0)
		##checar aqui no encuentra la funcion
		#norm_hessian_reg_output2 = normriemannianhessianmatrix_greylevel(output2,self.beta,self.dtype)
		norm_hessian_reg_output2 =regularizer1
		regularizer2 = torch.unsqueeze(norm_hessian_reg_output2,dim=0)

		return output, regularizer1, regularizer2


class OptimalFirstOrderDifferentialOperatorInvarianceGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, u0, u0x,u0y, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(OptimalFirstOrderDifferentialOperatorInvarianceGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.u0 = u0
		self.u0x = u0x
		self.u0y = u0y
		
		self.dtype=dtype	


	def forward(self, input,a):	

		output = self.net(input) #[1x1xHxW]
		output2 = torch.squeeze(output) #[HxW]

		output2_x = derivative_central_x_greylevel(output2,self.dtype)
		output2_y = derivative_central_y_greylevel(output2,self.dtype)	


		operator = torch.mul(torch.squeeze(torch.narrow(a,0,0,1)),output2_x-self.u0x) + torch.mul(torch.squeeze(torch.narrow(a,0,1,1)),output2_y-self.u0y) + torch.mul(torch.squeeze(torch.narrow(a,0,2,1)),output2 -self.u0)
		norm_operator = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype) + torch.square(operator))

		regularizer = torch.unsqueeze(norm_operator,dim=0)

		return output, regularizer


class OptimalFirstOrderDifferentialOperatorGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(OptimalFirstOrderDifferentialOperatorGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth

		
		self.dtype=dtype	


	def forward(self, input,a):	

		output = self.net(input) #[1x1xHxW]
		output2 = torch.squeeze(output) #[HxW]

		output2_x = derivative_central_x_greylevel(output2,self.dtype)
		output2_y = derivative_central_y_greylevel(output2,self.dtype)	

		a1 = torch.squeeze(torch.narrow(a,0,0,1))
		a2 = torch.squeeze(torch.narrow(a,0,1,1))
		a3 = torch.squeeze(torch.narrow(a,0,2,1))


		#operator = torch.mul(torch.squeeze(torch.narrow(a,0,0,1)),output2_x) + torch.mul(torch.squeeze(torch.narrow(a,0,1,1)),output2_y) + torch.mul(torch.squeeze(torch.narrow(a,0,2,1)),output2)
		norm_operator = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype) + torch.square(torch.mul(a1,output2_x)+torch.mul(a2,output2_y)+torch.mul(a3,output2)))   
		regularizer = torch.unsqueeze(norm_operator,dim=0)

		return output, regularizer


class OptimalSecondOrderDifferentialOperatorGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(OptimalSecondOrderDifferentialOperatorGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth

		
		self.dtype=dtype	


	def forward(self, input,a):	

		output = self.net(input) #[1x1xHxW]
		output2 = torch.squeeze(output) #[HxW]

		output2_x = derivative_central_x_greylevel(output2,self.dtype)
		output2_y = derivative_central_y_greylevel(output2,self.dtype)	
		output2_xx = derivative_xx_greylevel(output2,self.dtype)
		output2_xy = derivative_xy_greylevel(output2,self.dtype)
		output2_yy = derivative_yy_greylevel(output2,self.dtype)

		a1 = torch.squeeze(torch.narrow(a,0,0,1))
		a2 = torch.squeeze(torch.narrow(a,0,1,1))
		a3 = torch.squeeze(torch.narrow(a,0,2,1))
		a4 = torch.squeeze(torch.narrow(a,0,3,1))
		a5 = torch.squeeze(torch.narrow(a,0,4,1))
		a6 = torch.squeeze(torch.narrow(a,0,5,1))

		norm_operator = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype) + torch.square(torch.mul(a1,output2_xx)+torch.mul(a2,output2_xy)+torch.mul(a3,output2_yy)+torch.mul(a4,output2_x)+torch.mul(a5,output2_y)+torch.mul(a6,output2)))   
		regularizer = torch.unsqueeze(norm_operator,dim=0)

		return output, regularizer




class EuclideanDiracGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(EuclideanDiracGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]
		output2 = torch.squeeze(output) #[HxW]

		term1 = derivative_central_x_greylevel(output2,self.dtype)
		term2 = - derivative_central_y_greylevel(output2,self.dtype)
		term3 = 0.5*(derivative_xx_greylevel(output2,self.dtype))
		term4 = 0.5*(derivative_yy_greylevel(output2,self.dtype))
		norm = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype)+torch.square(term1)+torch.square(term2)+torch.square(term3)+torch.square(term4))
		norm = torch.unsqueeze(norm,dim=0)

		return output,norm



class RiemannianDiracGreyLevel(nn.Module):
	def __init__(self,input_depth, dtype, pad, beta, height, width, upsample_mode, n_channels=1, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(RiemannianDiracGreyLevel,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype
		self.beta = beta

	def forward(self, input):	

		output = self.net(input) #[1x1xHxW]
		output2 = torch.squeeze(output) #[HxW]

		g11,g12,g22 = riemannianmetric_greylevel(output2,self.beta,self.dtype)
		##checar aqui no encuentra la funcion
		#a,b,c,d = orthogonal_moving_frame(g11,g12,g22,self.dtype)
		a,b,c,d = 1,0,3,1	
		'''detframe = torch.mul(a,d) - torch.mul(b,c)

		invdetframe = torch.div(torch.ones((self.height,self.width)).type(self.dtype),detframe)
		invframe11 = torch.mul(invdetframe,d)
		invframe12 = - torch.mul(detframe,c)
		invframe21 = - torch.mul(detframe,b)	
		invframe22 = torch.mul(detframe,a)'''

		output2_x = derivative_central_x_greylevel(output2,self.dtype)
		output2_y = derivative_central_y_greylevel(output2,self.dtype)

		output2_xx = derivative_xx_greylevel(output2,self.dtype)
		output2_xy = derivative_xy_greylevel(output2,self.dtype)
		output2_yy = derivative_yy_greylevel(output2,self.dtype)

		term1 = torch.mul(a,output2_x) + torch.mul(b,output2_y)
		term2 = torch.mul(c,output2_x) + torch.mul(d,output2_y)
		term3 = 0.5*(torch.mul(torch.square(a),output2_xx)+2.*torch.mul(torch.mul(a,b),output2_xy)+torch.mul(torch.square(b),output2_yy))
		term4 = 0.5*(torch.mul(torch.square(c),output2_xx)+2.*torch.mul(torch.mul(c,d),output2_xy)+torch.mul(torch.square(d),output2_yy))

		norm = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype)+torch.square(term1)+torch.square(term2)+torch.square(term3)+torch.square(term4))
		norm = torch.unsqueeze(norm,dim=0)

		return output,norm 


class EuclideanDiracColor(nn.Module):
	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(EuclideanDiracColor,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x3xHxW]
		output2 = torch.squeeze(output) #[3xHxW]

		red_x = derivative_central_x_greylevel(output2[0,:,:],self.dtype)
		red_y = derivative_central_y_greylevel(output2[0,:,:],self.dtype)
		red_xx = derivative_xx_greylevel(output2[0,:,:],self.dtype)
		red_yy = derivative_yy_greylevel(output2[0,:,:],self.dtype)

		green_x = derivative_central_x_greylevel(output2[1,:,:],self.dtype)
		green_y = derivative_central_y_greylevel(output2[1,:,:],self.dtype)
		green_xx = derivative_xx_greylevel(output2[1,:,:],self.dtype)
		green_yy = derivative_yy_greylevel(output2[1,:,:],self.dtype)

		blue_x = derivative_central_x_greylevel(output2[2,:,:],self.dtype)
		blue_y = derivative_central_y_greylevel(output2[2,:,:],self.dtype)
		blue_xx = derivative_xx_greylevel(output2[2,:,:],self.dtype)
		blue_yy = derivative_yy_greylevel(output2[2,:,:],self.dtype)


		norm = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype)+torch.square(red_x+green_y)+torch.square(green_x-red_y) +torch.square(0.5*red_xx)+torch.square(0.5*red_yy)+torch.square(0.5*green_xx)+torch.square(0.5*green_yy) //
			+torch.square(blue_x+red_y)+torch.square(red_x-blue_y) +torch.square(0.5*blue_xx)+torch.square(0.5*blue_yy)+torch.square(0.5*red_xx)+torch.square(0.5*red_yy) //
			+torch.square(green_x+blue_y)+torch.square(blue_x-green_y) +torch.square(0.5*green_xx)+torch.square(0.5*green_yy)+torch.square(0.5*blue_xx)+torch.square(0.5*blue_yy))
		norm = torch.unsqueeze(norm,dim=0)

		return output,norm
	
	
class EuclideanDiracColor2(nn.Module):
	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(EuclideanDiracColor2,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x3xHxW]
		output2 = torch.squeeze(output) #[3xHxW]

		red_x = derivative_central_x_greylevel(output2[0,:,:],self.dtype)
		red_y = derivative_central_y_greylevel(output2[0,:,:],self.dtype)
		norm_grad_red_squared = torch.square(red_x) + torch.square(red_y)
		green_x = derivative_central_x_greylevel(output2[1,:,:],self.dtype)
		green_y = derivative_central_y_greylevel(output2[1,:,:],self.dtype)
		norm_grad_green_squared = torch.square(green_x) + torch.square(green_y)
		blue_x = derivative_central_x_greylevel(output2[2,:,:],self.dtype)
		blue_y = derivative_central_y_greylevel(output2[2,:,:],self.dtype)
		norm_grad_blue_squared = torch.square(blue_x) + torch.square(blue_y)


		norm = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype) + torch.maximum(torch.zeros((self.height,self.width)).type(self.dtype),norm_grad_red_squared + norm_grad_green_squared + norm_grad_blue_squared //
			+ (torch.mul(red_x,green_y)-torch.mul(red_y,green_x)) + (torch.mul(blue_x,red_y)-torch.mul(blue_y,red_x)) + (torch.mul(green_x,blue_y)-torch.mul(green_y,blue_x)))) 
		norm = torch.unsqueeze(norm,dim=0)

		return output,norm	
	
class EuclideanDiracColorRGB_beta(nn.Module):
	def __init__(self,input_depth, dtype, pad, beta, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(EuclideanDiracColorRGB_beta,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype
		self.beta = beta

	def forward(self, input):	

		output = self.net(input) #[1x3xHxW]
		output2 = torch.squeeze(output) #[3xHxW]

		red_x = derivative_central_x_greylevel(output2[0,:,:],self.dtype)
		red_y = derivative_central_y_greylevel(output2[0,:,:],self.dtype)
		norm_grad_red_squared = torch.square(red_x) + torch.square(red_y)
		green_x = derivative_central_x_greylevel(output2[1,:,:],self.dtype)
		green_y = derivative_central_y_greylevel(output2[1,:,:],self.dtype)
		norm_grad_green_squared = torch.square(green_x) + torch.square(green_y)
		blue_x = derivative_central_x_greylevel(output2[2,:,:],self.dtype)
		blue_y = derivative_central_y_greylevel(output2[2,:,:],self.dtype)
		norm_grad_blue_squared = torch.square(blue_x) + torch.square(blue_y)


		norm = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype) + torch.maximum(torch.zeros((self.height,self.width)).type(self.dtype),(1.+self.beta**2)*(norm_grad_red_squared + norm_grad_green_squared + norm_grad_blue_squared) + 2.*self.beta*((torch.mul(red_x,green_y)-torch.mul(red_y,green_x)) + (torch.mul(blue_x,red_y)-torch.mul(blue_y,red_x)) + (torch.mul(green_x,blue_y)-torch.mul(green_y,blue_x))))) 
		norm = torch.unsqueeze(norm,dim=0)

		return output,norm	
	
class UnknownOperatorV4(nn.Module):
	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(UnknownOperatorV4,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x3xHxW]
		output2 = torch.squeeze(output) #[3xHxW]

		red_x = derivative_central_x_greylevel(output2[0,:,:],self.dtype)
		red_y = derivative_central_y_greylevel(output2[0,:,:],self.dtype)
		norm_grad_red_squared = torch.square(red_x) + torch.square(red_y)
		green_x = derivative_central_x_greylevel(output2[1,:,:],self.dtype)
		green_y = derivative_central_y_greylevel(output2[1,:,:],self.dtype)
		norm_grad_green_squared = torch.square(green_x) + torch.square(green_y)
		blue_x = derivative_central_x_greylevel(output2[2,:,:],self.dtype)
		blue_y = derivative_central_y_greylevel(output2[2,:,:],self.dtype)
		norm_grad_blue_squared = torch.square(blue_x) + torch.square(blue_y)


		norm = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype) + norm_grad_red_squared + torch.abs(torch.mul(green_x,blue_y)-torch.mul(green_y,blue_x)) + norm_grad_green_squared + torch.abs(torch.mul(blue_x,red_y)-torch.mul(blue_y,red_x)) + norm_grad_blue_squared + torch.abs(torch.mul(red_x,green_y)-torch.mul(red_y,green_x))) 
		norm = torch.unsqueeze(norm,dim=0)

		return output,norm	

class UnknownOperatorV4OPP(nn.Module):
	def __init__(self,input_depth, dtype, pad, height, width, upsample_mode, n_channels=3, act_fun='LeakyReLU', skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5, downsample_mode='stride'):
		super(UnknownOperatorV4OPP,self).__init__()
		self.net = skip(input_depth, n_channels, num_channels_down = [skip_n33d]*num_scales if isinstance(skip_n33d, int) else skip_n33d,
                                            num_channels_up =   [skip_n33u]*num_scales if isinstance(skip_n33u, int) else skip_n33u,
                                            num_channels_skip = [skip_n11]*num_scales if isinstance(skip_n11, int) else skip_n11, 
                                            upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                            need_sigmoid=True, need_bias=True, pad=pad, act_fun=act_fun)

		self.height=height
		self.width=width
		self.input_depth = input_depth
		self.dtype=dtype

	def forward(self, input):	

		output = self.net(input) #[1x3xHxW]
		output2 = torch.squeeze(output) #[3xHxW]

		lum = (1./(0.6*sqrt(3)))*(output2[0,:,:]+output2[1,:,:]+output2[2,:,:])
		chrom1 = (1./sqrt(2))*(output2[0,:,:]-output2[1,:,:])
		chrom2 = (1/sqrt(6))*(output2[0,:,:]+output2[1,:,:]-2*output2[2,:,:])

		lum_x = derivative_central_x_greylevel(lum,self.dtype)
		lum_y = derivative_central_y_greylevel(lum,self.dtype)
		norm_grad_lum_squared = torch.square(lum_x) + torch.square(lum_y)

		chrom1_x = derivative_central_x_greylevel(chrom1,self.dtype)
		chrom1_y = derivative_central_y_greylevel(chrom1,self.dtype)
		norm_grad_chrom1_squared = torch.square(chrom1_x) + torch.square(chrom1_y)

		chrom2_x = derivative_central_x_greylevel(chrom2,self.dtype)
		chrom2_y = derivative_central_y_greylevel(chrom2,self.dtype)
		norm_grad_chrom2_squared = torch.square(chrom2_x) + torch.square(chrom2_y)


		norm = torch.sqrt(0.00001*torch.ones((self.height,self.width)).type(self.dtype) + norm_grad_lum_squared + torch.abs(torch.mul(chrom1_x,chrom2_y)-torch.mul(chrom1_y,chrom2_x)) + norm_grad_chrom1_squared + torch.abs(torch.mul(chrom2_x,lum_y)-torch.mul(chrom2_y,lum_x)) + norm_grad_chrom2_squared + torch.abs(torch.mul(lum_x,chrom1_y)-torch.mul(lum_y,chrom1_x))) 
		norm = torch.unsqueeze(norm,dim=0)

		return output,norm