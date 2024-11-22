import sys
import math
import numpy as np
from .MS9_dmpad_hgf_ar1_lvl3_transp import MS9_dmpad_hgf_ar1_lvl3_transp
from ...code_inversion.tapas_sgm import tapas_sgm


def MS9_dmpad_hgf_ar1_lvl3(r, p, varargin = None) :
	'''
	Calculates the trajectories of the agent's representations under the HGF

	This function can be called in two ways:

	(1) hgf_binary_ar1(r, p)

		where r is the structure generated by fitModel and p is the parameter vector in native space;

	(2) hgf_binary_ar1(r, ptrans, 'trans')

		where r is the structure generated by fitModel, ptrans is the parameter vector in
		transformed space, and 'trans' is a flag indicating this.

	--------------------------------------------------------------------------------------------------
	Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ
	modified Andreea Diaconescu, TNU, UZH & ETHZ 22/04/2013

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''


	# Transform paramaters back to their native space if needed
	if varargin != None and varargin == 'trans' :
		[p, _] = MS9_dmpad_hgf_ar1_lvl3_transp(r, p)

	# Number of levels
	try :
		l = r['c_prc'].n_levels
	except :
		l = (p.shape[1]+1)/6
		if l != math.floor(l) :
			sys.exit('Cannot determine number of levels')

	# Unpack parameters
	mu_0 = p[0:1,0:l]
	sa_0 = p[0:1,l:2*l]
	phi  = p[0:1,2*l:3*l]
	m    = p[0:1,3*l:4*l]
	ka   = p[0:1,4*l:5*l-1]
	om   = p[0:1,5*l-1:6*l-2]
	th   = p[0:1,6*l-2]

	# Add dummy "zeroth" trial
	u = np.concatenate([np.zeros((1,1)), r['u'][:,0:1]], axis = 0)

	# Number of trials (including prior)
	n = u.shape[0]

	# Assume that if u has more than one column, the last contains t
	try :
		if r['c_prc'].irregular_intervals :
			if u.shape[1] > 1 :
				t = np.concatenate([np.zeros((1,1)), r['u'][:,-1:]], axis = 0)
			else :
				sys.exit('Error: input matrix must contain more than one column if irregular_intervals is set to true.')
		else :
			t = np.ones((n,1))
	except :
		if u.shape[1] > 1 :
			t = np.concatenate([np.zeros((1,1)), r['u'][:,-1:]], axis = 0)
		else :
			t = np.ones((n,1))

	# Initialize updated quantities

	# Representations
	mu    = np.empty((n,l))
	pi    = np.empty((n,l))
	mu[:] = np.nan
	pi[:] = np.nan

	# Other quantities
	muhat    = np.empty((n,l))
	pihat    = np.empty((n,l))
	w        = np.empty((n,l-1))
	da       = np.empty((n,l))
	muhat[:] = np.nan
	pihat[:] = np.nan
	w    [:] = np.nan
	da   [:] = np.nan

	# Representation priors
	# Note: first entries of the other quantities remain
	# NaN because they are undefined and are thrown away
	# at the end; their presence simply leads to consistent
	# trial indices.
	mu[0,0]  = tapas_sgm(mu_0[0,0], 1.)
	pi[0,0]  = np.inf
	mu[0,1:] = mu_0[0,1:]
	pi[0,1:] = 1./sa_0[0,1:]

	# Representation update loop
	# Pass through trials
	for k in range(1,n) : # k = 2:1:n
		if not k-1 in r['ign'] : # if not(ismember(k-1, r.ign))
			
			######################
			# Effect of input u(k)
			######################
			
			# second level prediction
			muhat[k,1] = mu[k-1,1] + t[k,0] * phi[0,1] * (m[0,1] - mu[k-1,1])
			
			# 1st level
			# ~~~~~~~~~
			# Prediction
			muhat[k,0] = tapas_sgm(muhat[k,1], 1.)
			
			# Precision of prediction
			pihat[k,0] = 1./(muhat[k,0] * (1.-muhat[k,0]))
			
			# Updates
			pi[k,0] = np.inf
			mu[k,0] = u[k,0]
			
			# Prediction error
			da[k,0] = mu[k,0] - muhat[k,0]
			
			# Second level
			# ~~~~~~~~~~~~
			# see above for prediction
			
			# Precision of prediction
			pihat[k,1] = 1./(1./pi[k-1,1] + np.exp(ka[0,1] * mu[k-1,2] + om[0,1]))
			
			# Updates 
			pi[k,1] = pihat[k,1] + 1./pihat[k,0]
			mu[k,1] = muhat[k,1] + 1./pi[k,1]*da[k,0] 
			
			# Volatility prediction error  
			da[k,1] = (1./pi[k,1] + (mu[k,1] - muhat[k,1])**2) * pihat[k,1] - 1.
			 
			if l > 3 :
				# Pass through higher levels
				# ~~~~~~~~~~~~~~~~~~~~~~~~~~
				for j in range(2,l-1) : # j = 3:l-1
					# Prediction
					muhat[k,j] = mu[k-1,j] + t[k,0] * phi[0,j] * (m[0,j] - mu[k-1,j])
					
					# Precision of prediction
					pihat[k,j] = 1./(1./pi[k-1,j] + t[k,0] * np.exp(ka[0,j] * mu[k-1,j+1] + om[0,j]))
					
					# Weighting factor
					w[k,j-1] = t[k,0] * np.exp(ka[0,j-1] * mu[k-1,j] + om[0,j-1]) * pihat[k,j-1]
					
					# Updates
					pi[k,j] = pihat[k,j] + 0.5*ka[0,j-1]**2 * w[k,j-1] * (w[k,j-1] + (2.*w[k,j-1]-1.) * da[k,j-1])
					
					if pi[k,j] <= 0 :
						sys.exit('Error: negative posterior precision. Parameters are in a region where model assumptions are violated.')
					
					mu[k,j] = muhat[k,j] + 0.5/pi[k,j] * ka[0,j-1] * w[k,j-1] * da[k,j-1]
					
					# Volatility prediction error
					da[k,j] = (1./pi[k,j] + (mu[k,j] - muhat[k,j])**2) * pihat[k,j] - 1.
			
			# Last level
			# ~~~~~~~~~~
			# Prediction
			muhat[k,l-1] = mu[k-1,l-1] + t[k,0] * phi[0,l-1] * (m[0,l-1] - mu[k-1,l-1])
			
			# Precision of prediction
			pihat[k,l-1] = 1./(1./pi[k-1,l-1] + t[k,0] * th)
			
			# Weighting factor
			w[k,l-2] = t[k,0] * np.exp(ka[0,l-2] * mu[k-1,l-1] + om[0,l-2]) * pihat[k,l-2]
			
			# Updates
			pi[k,l-1] = pihat[k,l-1] + 0.5*ka[0,l-2]**2 * w[k,l-2] * (w[k,l-2] + (2.*w[k,l-2] - 1.) * da[k,l-2])
			
			if pi[k,l-1] <= 0 :
				sys.exit('Error: negative posterior precision. Parameters are in a region where model assumptions are violated.')
			
			mu[k,l-1] = muhat[k,l-1] + 0.5/pi[k,l-1] * ka[0,l-2] * w[k,l-2] * da[k,l-2]
			
			# Volatility prediction error
			da[k,l-1] = (1./pi[k,l-1] + (mu[k,l-1] - muhat[k,l-1])**2) * pihat[k,l-1] - 1.

		else :
			mu[k,:] = mu[k-1,:]
			pi[k,:] = pi[k-1,:]
			
			muhat[k,:] = muhat[k-1,:]
			pihat[k,:] = pihat[k-1,:]
			
			w [k,:] = w [k-1,:]
			da[k,:] = da[k-1,:]

	# Remove representation priors
	mu = np.delete(mu, 0, axis = 0)
	pi = np.delete(pi, 0, axis = 0)

	# Remove other dummy initial values
	muhat = np.delete(muhat, 0, axis = 0)
	pihat = np.delete(pihat, 0, axis = 0)
	w     = np.delete(w    , 0, axis = 0)
	da    = np.delete(da   , 0, axis = 0)

	# Create result data structure
	traj = {}

	traj['mu'   ] = mu
	traj['sa'   ] = 1./pi

	traj['muhat'] = muhat
	traj['sahat'] = 1./pihat

	traj['w'    ] = w
	traj['da'   ] = da

	# Create matrices for use by observation model
	infStates        = np.empty((n-1,l,2))
	infStates[:]     = np.nan
	infStates[:,:,0] = traj['muhat']
	infStates[:,:,1] = traj['sahat']

	return [traj, infStates]