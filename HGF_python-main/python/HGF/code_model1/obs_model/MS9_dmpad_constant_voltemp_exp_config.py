import numpy as np
import math
from ...code_inversion.tapas_logit import tapas_logit
from .MS9_dmpad_constant_voltemp_exp import MS9_dmpad_constant_voltemp_exp
from .MS9_dmpad_constant_voltemp_exp_transp import MS9_dmpad_constant_voltemp_exp_transp


class MS9_dmpad_constant_voltemp_exp_config(object) :
	'''
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	Contains the configuration for the IOIO constant weight observation model with mu3hat as
	decision temperature

	--------------------------------------------------------------------------------------------------
	Copyright (C) 2012 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	def __init__(self, ptrans = None) :

		# Model name
		self.model = 'MS9_dmpad_constant_voltemp_exp'

		# Sufficient statistics of Gaussian parameter priors

		if ptrans == None :
			# Zeta_1
			self.logitze1mu = tapas_logit(0.5,1.) 
			self.logitze1sa = 1.
			
			# Zeta_2
			self.logze2mu = math.log(48.)
			self.logze2sa = 1.
		else :
			# Zeta_1
			self.logitze1mu = ptrans[0,0] # 0;
			self.logitze1sa = ptrans[0,1]
			
			# Zeta_2
			self.logze2mu = ptrans[0,2]
			self.logze2sa = ptrans[0,3]

		# Gather prior settings in vectors
		self.priormus = np.array([[
			self.logitze1mu,
			self.logze2mu
			]])

		self.priorsas = np.array([[
			self.logitze1sa,
			self.logze2sa
			]])


		# Model filehandle
		self.obs_fun = MS9_dmpad_constant_voltemp_exp

		# Handle to function that transforms observation parameters to their native space
		# from the space they are estimated in
		self.transp_obs_fun = MS9_dmpad_constant_voltemp_exp_transp

		return
