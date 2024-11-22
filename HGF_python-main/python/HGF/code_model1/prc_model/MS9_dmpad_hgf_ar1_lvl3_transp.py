import numpy as np
from ...code_inversion.tapas_sgm import tapas_sgm


def MS9_dmpad_hgf_ar1_lvl3_transp(r, ptrans) :
	'''
	--------------------------------------------------------------------------------------------------
	Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''


	pvec    = np.empty(ptrans.shape)
	pvec[:] = np.nan
	pstruct = {}

	l = r['c_prc'].n_levels

	pvec[0,0:l]         = ptrans[0,0:l]                               # mu_0
	pstruct['mu_0']     = pvec[0:1,0:l]
	pvec[0,l:2*l]       = np.exp(ptrans[0,l:2*l])                     # sa_0
	pstruct['sa_0']     = pvec[0:1,l:2*l]
	pvec[0,2*l:3*l]     = tapas_sgm(ptrans[0,2*l:3*l],1.)             # phi
	pstruct['phi']      = pvec[0:1,2*l:3*l]
	pvec[0,3*l:4*l]     = ptrans[0,3*l:4*l]                           # m
	pstruct['m']        = pvec[0:1,3*l:4*l]
	pvec[0,4*l:5*l-1]   = tapas_sgm(ptrans[0,4*l:5*l-1],r['c_prc'].kaub) # ka
	pstruct['ka']       = pvec[0:1,4*l:5*l-1]
	pvec[0,5*l-1:6*l-2] = ptrans[0,5*l-1:6*l-2]                       # om
	pstruct['om']       = pvec[0:1,5*l-1:6*l-2]
	pvec[0,6*l-2]       = tapas_sgm(ptrans[0,6*l-2],r['c_prc'].thub)     # th
	pstruct['th']       = pvec[0,6*l-2]

	return [pvec, pstruct]
