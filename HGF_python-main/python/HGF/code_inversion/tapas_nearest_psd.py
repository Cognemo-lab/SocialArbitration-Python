import numpy as np


def tapas_nearest_psd(X) :
	'''
	Finds the nearest positive semi-defnite matrix to X

	--------------------------------------------------------------------------------------------------
	Copyright (C) 2020 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Input: X - a square matrix

	Output: X - nearest positive semi-definite matrix to input X

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	# Ensure symmetry
	X = (X.T + X)/2.

	# Continue until X is positive semi-definite
	while (np.linalg.eigvals(X) < 0).any() :
		# V: right eigenvectors, D: diagonalized X (X*V = V*D <=> X = V*D*V')
		D, V = np.linalg.eig(X)
		D = np.diag(D)
		# Replace negative eigenvalues with 0 in D
		D = np.maximum(0., D)
		# Transform back
		X = np.matmul(V, np.matmul(D, V.T))
		# Ensure symmetry
		X = (X.T + X)/2.

	return X
