import numpy as np
from copy import deepcopy
from .tapas_riddersdiff import tapas_riddersdiff


def tapas_riddersgradient(f, x,
		init_h     = 1,
		div        = 1.2,
		min_steps  = 3,
		max_steps  = 100,
		tf         = 2
		) :
	'''
	Calculates the gradient of the function f at point x according to Ridders' method:

	Ridders, CJF. (1982). Accurate computation of F'(x) and F'(x) F''(x). Advances in Engineering
	    Software, 4(2), 75-6.

	INPUT:
	   f             Function handle of a real function of n real variables which are passed as
	                 *one* vector with n elements
	   x             Point at which to differentiate f

	OUTPUT:
	   gradf         Gradient of f at x (row vector)
	   err           Error estimates (row vector)

	OPTIONS:
	   Optionally, the third argument of the function can be a structure containing further
	   settings for Ridder's method.

	   init_h        Initial finite difference (default: 1)
	   div           Divisor used to reduce h on each step (default: 1.2)
	   min_steps     Minimum number of steps in h (default: 3)
	   max_steps     Maximum number of steps in h (default: 100)
	   tf            Terminate if last step worse than preceding by a factor of tf
	                 (default: 2)

	--------------------------------------------------------------------------------------------------
	Copyright (C) 2012-2013 Christoph Mathys, TNU, UZH & ETHZ

	This file is released under the terms of the GNU General Public Licence (GPL), version 3. You can
	redistribute it and/or modify it under the terms of the GPL (either version 3 or, at your option,
	any later version). For further details, see the file COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	# Convert x to one-dimensional float type array
	xflat = x.flatten().astype(float)
	n = len(xflat)

	# Defaults
	gradf              = np.empty(n)
	err                = np.empty(n)
	gradf[:]           = np.nan
	err[:]             = np.nan

	# Check if f and x match
	try :
		f(xflat)
	except :
		print('tapas:hgf:ridders:CannotEvalFun', 'Function cannot be evaluated at differentiation point.');

	# Loop through argument variables
	for i in range(n) :

		# Construct filehandle to be passed to riddersdiff
		fxih = lambda xi : fxi(f,xflat,i,xi)

		# Calculate derivative
		[gradf[i], err[i]] = tapas_riddersdiff(fxih,xflat[i],
				init_h     = init_h   ,
				div        = div      ,
				min_steps  = min_steps,
				max_steps  = max_steps,
				tf         = tf       )

	# Return row vectors
	gradf.shape = (1,n)
	err  .shape = (1,n)
	return [gradf, err]


def fxi(f,x,i,xi) :
	xx    = deepcopy(x)
	xx[i] = xi
	fxi   = f(xx)
	return fxi
