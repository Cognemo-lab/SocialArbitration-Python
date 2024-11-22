import numpy as np
from ...code_inversion.tapas_sgm import tapas_sgm


def MS9_dmpad_constant_voltemp_exp(r, infStates, ptrans) :
	'''
	Calculates the log-probability of response y=1 under the IOIO response model with constant
	weight zeta_1

	--------------------------------------------------------------------------------------------------
	Copyright (C) 2012 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	######## Transform zetas to their native space ###############
	ze1 = tapas_sgm(ptrans[0,0],1.)
	ze2 = ptrans[0,1]

	# Initialize returned log-probabilities as NaNs so that NaN is
	# returned for all irregualar trials
	logp    = np.empty((max(infStates.shape),1))
	logp[:] = np.nan

	# Weed irregular trials out from inferred states, cue inputs, and responses
	mu1hat = infStates[:,0:1,0]
	mu1hat = np.delete(mu1hat, r['irr'], axis = 0)

	mu3hat = infStates[:,2:3,0]
	mu3hat = np.delete(mu3hat, r['irr'], axis = 0)

	c = r['u'][:,1:2]
	c = np.delete(c, r['irr'], axis = 0)

	y = r['y'][:,0:1]
	y = np.delete(y, r['irr'], axis = 0)

	# Belief vector
	b = ze1*mu1hat + (1.-ze1)*c
	beta = np.exp(-mu3hat+ze2)

	# Calculate log-probabilities for non-irregular trials
	logp[[idx for idx in range(logp.shape[0]) if idx not in r['irr']],0:1] = y*beta*np.log(b/(1.-b)) + np.log((1.-b)**(beta) / ((1.-b)**(beta) + b**(beta)))

	return logp
