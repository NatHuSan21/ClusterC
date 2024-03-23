def Levi_Cevita_connection_coefficients_standard_frame(g11,g12,g22):

	detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[1x1xHxW]

	invdetg = torch.div(torch.ones((1,1,self.height,self.width)).type(torch.cuda.FloatTensor),detg) #[1x1xHxW]
	invg11 = torch.mul(invdetg,g22) #[1x1xHxW]
	invg12 = - torch.mul(invdetg,g12) #[1x1xHxW]
	invg22 = torch.mul(invdetg,g11)	#[1x1xHxW]


	differential_g11 = kornia.filters.SpatialGradient()(g11) # [1x1x2xHxW]
	g11_x = torch.narrow(differential_g11,2,0,1) # [1x1x1xHxW]	
	g11_x = torch.squeeze(g11_x,dim=2) # [1x1xHxW]
	g11_y = torch.narrow(differential_g11,2,1,1) # [1x1x1xHxW]
	g11_y = torch.squeeze(g11_y,dim=2) # [1x1xHxW]

	differential_g12 = kornia.filters.SpatialGradient()(g12) # [1x1x2xHxW]
	g12_x = torch.narrow(differential_g12,2,0,1) # [1x1x1xHxW]	
	g12_x = torch.squeeze(g12_x,dim=2) # [1x1xHxW]
	g12_y = torch.narrow(differential_g12,2,1,1) # [1x1x1xHxW]
	g12_y = torch.squeeze(g12_y,dim=2) # [1x1xHxW]

	differential_g22 = kornia.filters.SpatialGradient()(g22) # [1x1x2xHxW]
	g22_x = torch.narrow(differential_g22,2,0,1) # [1x1x1xHxW]	
	g22_x = torch.squeeze(g22_x,dim=2) # [1x1xHxW]
	g22_y = torch.narrow(differential_g22,2,1,1) # [1x1x1xHxW]
	g22_y = torch.squeeze(g22_y,dim=2) # [1x1xHxW]

	Gamma1_11 = torch.mul(0.5*invdetg,torch.mul(g22,g11_x)-torch.mul(g12,2.*g12_x-g11_y)) # [1x1xHxW]
	Gamma2_22 = torch.mul(0.5*invdetg,torch.mul(g11,g22_y)-torch.mul(g12,2.*g12_y-g22_x)) # [1x1xHxW]
	Gamma1_12 = torch.mul(0.5*invdetg,torch.mul(g22,g11_y)-torch.mul(g12,g22_x)) # [1x1xHxW]
	Gamma1_21 = Gamma1_12 # [1x1xHxW]
	Gamma2_12 = torch.mul(0.5*invdetg,torch.mul(g11,g22_x)-torch.mul(g12,g11_y)) # [1x1xHxW]
	Gamma2_21 = Gamma2_12 # [1x1xHxW]
	Gamma1_22 = torch.mul(0.5*invdetg,torch.mul(g22,2.*g12_y-g22_x)-torch.mul(g12,g22_y)) # [1x1xHxW]
	Gamma2_11 = torch.mul(0.5*invdetg,torch.mul(g11,2.*g12_x-g11_y)-torch.mul(g12,g11_x)) # [1x1xHxW]

	return Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11




def Levi_Cevita_connection_coefficients_orthogonal_frame(g11,g12,g22,a,b,c,d):

	Gamma1_11,Gamma2_22,Gamma1_12,Gamma1_21,Gamma2_12, Gamma2_21, Gamma1_22, Gamma2_11 = Levi_Cevita_connection_coefficients_standard_frame(g11,g12,g22)
	differential_a = kornia.filters.SpatialGradient()(a) # [1x1x2xHxW]
	a_x = torch.narrow(differential_a,2,0,1) # [1x1x1xHxW]
	a_x = torch.squeeze(a_x,dim=2) # [1x1xHxW]
	a_y = torch.narrow(differential_a,2,1,1) # [1x1x1xHxW]
	a_y = torch.squeeze(a_y,dim=2) # [1x1xHxW]
	b_x = torch.narrow(differential_b,2,0,1) # [1x1x1xHxW]
	b_x = torch.squeeze(b_x,dim=2) # [1x1xHxW]
	b_y = torch.narrow(differential_b,2,1,1) # [1x1x1xHxW]
	b_y = torch.squeeze(b_y,dim=2) # [1x1xHxW]
	c_x = torch.narrow(differential_c,2,0,1) # [1x1x1xHxW]
	c_x = torch.squeeze(c_x,dim=2) # [1x1xHxW]
	c_y = torch.narrow(differential_c,2,1,1) # [1x1x1xHxW]
	c_y = torch.squeeze(c_y,dim=2) # [1x1xHxW]
	d_x = torch.narrow(differential_d,2,0,1) # [1x1x1xHxW]
	d_x = torch.squeeze(d_x,dim=2) # [1x1xHxW]
	d_y = torch.narrow(differential_d,2,1,1) # [1x1x1xHxW]
	d_y = torch.squeeze(d_y,dim=2) # [1x1xHxW]


	Gamma2_12prim = torch.mul(torch.div(torch.ones((1,1,self.height,self.width)).type(torch.cuda.FloatTensor),torch.mul(a,d)-torch.mul(b,c)),-torch.mul(torch.mul(a,b),a_x) \
	-torch.mul(torch.mul(torch.square(a),b),Gamma1_11)  - 2.*torch.mul(torch.mul(a,torch.square(b)),Gamma1_12) - torch.mul(torch.square(b),a_y) - torch.mul(torch.pow(b,3),Gamma1_22) \
	+ torch.mul(torch.pow(a,3),Gamma2_11) + torch.mul(torch.square(a),b_x) + 2.*torch.mul(torch.mul(b,torch.square(a)),Gamma2_12) + torch.mul(torch.mul(a,b),b_y) + torch.mul(torch.mul(a,torch.square(b)),Gamma2_22))  #[1x1xHxW]
		
	Gamma2_21prim = torch.mul(torch.div(torch.ones((1,1,self.height,self.width)).type(torch.cuda.FloatTensor),torch.mul(a,d)-torch.mul(b,c)),-torch.mul(torch.mul(b,c),a_x) \
	-torch.mul(torch.mul(torch.mul(a,b),c),Gamma1_11)-torch.mul(torch.mul(torch.mul(b,c)+torch.mul(a,d),b),Gamma1_12) - torch.mul(torch.mul(b,d),a_y) - torch.mul(torch.mul(torch.square(b),d),Gamma1_22) \
	+torch.mul(torch.mul(torch.square(a),c),Gamma2_11) + torch.mul(torch.mul(a,c),b_x) + torch.mul(torch.mul(torch.mul(b,c)+torch.mul(a,d),a),Gamma2_12) + torch.mul(torch.mul(a,d),b_y) + torch.mul(torch.mul(torch.mul(a,b),d),Gamma2_22))


	return Gamma2_12prim,Gamma2_21prim


def Orthogonal_moving_frame(g11,g12,g22):

	detg = torch.mul(g11,g22) - torch.mul(g12,g12) #[1x1xHxW]

	lambda_plus = (1./2.)*(g11+g22+torch.sqrt(torch.square(g11+g22)-4.*detg))
	lambda_moins = (1./2.)*(g11+g22-torch.sqrt(torch.square(g11+g22)-4.*detg))

	a = torch.div(lambda_plus-g22,torch.mul(torch.sqrt(lambda_plus),torch.sqrt(torch.square(g12)+torch.square(lambda_plus-g22))))
	b = torch.div(g12,torch.mul(torch.sqrt(lambda_plus),torch.sqrt(torch.square(g12)+torch.square(lambda_plus-g22))))
	c = torch.mul(torch.sign(g12), torch.div(lambda_moins-g22,torch.mul(torch.sqrt(lambda_moins),torch.sqrt(torch.square(g12)+torch.square(lambda_moins-g22)))))
	d = torch.mul(torch.sign(g12), g12,torch.mul(torch.sqrt(lambda_moins),torch.sqrt(torch.square(g12)+torch.square(lambda_moins-g22))))

	return a,b,c,d


