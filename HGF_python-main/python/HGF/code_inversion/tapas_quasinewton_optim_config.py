from .tapas_quasinewton_optim import tapas_quasinewton_optim

class tapas_quasinewton_optim_config(object) :
	'''
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	Contains the configuration for the Broyden, Fletcher, Goldfarb and Shanno (BFGS)
	quasi-Newton optimization algorithm

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	def __init__(self) :

		# Algorithm name
		self.algorithm = 'BFGS quasi-Newton'

		# Verbosity
		self.verbose   = False

		# Options for optimization
		self.tolGrad   = 1e-3
		self.tolArg    = 1e-3
		self.maxStep   = 1
		self.maxIter   = 100
		self.maxRegu   = 16
		self.maxRst    = 10
		self.nRandInit = 0

		# Algorithm filehandle
		self.opt_algo  = tapas_quasinewton_optim
