import numpy as np
from ...code_inversion.tapas_sgm import tapas_sgm


def MS9_dmpad_constant_voltemp_exp_transp(r, ptrans) :
	'''
	--------------------------------------------------------------------------------------------------
	Copyright (C) 2012 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	pvec    = np.empty(ptrans.shape)
	pvec[:] = np.nan
	pstruct = {}
	# function that transforms observation parameters to their native space from the space they are estimated in

#	pvec[0,0]      = np.exp(ptrans[0,0])       # ze1
#	pstruct['ze1'] = pvec[0,0]
	pvec[0,0]      = tapas_sgm(ptrans[0,0],1.) # ze1
	pstruct['ze1'] = pvec[0,0]
	pvec[0,1]      = np.exp(ptrans[0,1])       # ze2
	pstruct['ze2'] = pvec[0,1]

	return [pvec, pstruct]
