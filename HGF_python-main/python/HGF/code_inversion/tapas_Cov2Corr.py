import sys
import numpy as np


def tapas_Cov2Corr(Cov) :
	'''
	Converts a covariance matrix into a correlation matrix

	--------------------------------------------------------------------------------------------------
	Copyright (C) 2013 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	# Check if Cov is symmetric
	if (Cov.T != Cov).any() :
		sys.exit('tapas:hgf:Cov2Corr:MatNotSymm - Input matrix is not symmetric.')

	# Check if Cov is positive semi-definite
	if np.isinf(Cov).any() or np.isnan(Cov).any() or (np.linalg.eigvals(Cov) < 0).any() :
		sys.exit('tapas:hgf:Cov2Corr:MatNotPosDef - Input matrix is not positive semi-definite.')

	sdev = np.sqrt(np.diag(Cov).reshape((-1,1)))
	Norm = np.matmul(sdev, sdev.T)
	Corr = Cov/Norm

	return Corr
