import numpy as np
from copy import deepcopy
from .tapas_riddersdiff2 import tapas_riddersdiff2
from .tapas_riddersdiffcross import tapas_riddersdiffcross


def tapas_riddershessian(f, x,
		init_h     = 1,
		div        = 1.2,
		min_steps  = 3,
		max_steps  = 100,
		tf         = 2
		) :
	'''
	Calculates the hessian of the function f at point x according to Ridders' method:

	Ridders, CJF. (1982). Accurate computation of F'(x) and F'(x) F''(x). Advances in Engineering
		Software, 4(2), 75-6.

	INPUT:
	   f             Function handle of a real function of n real variables which are passed as
					 *one* vector with n elements
	   x             Point at which to differentiate f

	OUTPUT:
	   hessf         Hessian of f at x
	   err           Error estimates

	OPTIONS:
	   Optionally, the third argument of the function can be a structure containing further
	   settings for Ridder's method.

	   varargin{1}.init_h      Initial finite difference (default: 1)
	   varargin{1}.div         Divisor used to reduce h on each step (default: 1.2)
	   varargin{1}.min_steps   Minimum number of steps in h (default: 3)
	   varargin{1}.max_steps   Maximum number of steps in h (default: 100)
	   varargin{1}.tf          Terminate if last step worse than preceding by a factor of tf
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

	hessf              = np.empty((n,n))
	err                = np.empty((n,n))
	hessf[:]           = np.nan
	err  [:]           = np.nan

	# Check if f and x match
	try :
		f(xflat)
	except :
		print('tapas:hgf:ridders:CannotEvalFun', 'Function cannot be evaluated at differentiation point.')

	# First: diagonal elements
	# Loop through argument variables
	for i in range(n) :

		# Construct filehandle to be passed to riddersdiff2
		fxih = lambda xi : fxi(f,xflat,i,xi)

		# Calculate derivative
		[hessf[i,i], err[i,i]] = tapas_riddersdiff2(fxih,xflat[i],
				init_h     = init_h   ,
				div        = div      ,
				min_steps  = min_steps,
				max_steps  = max_steps,
				tf         = tf       )

	# Second: off-diagonal elements
	# Loop through argument variables
	for i in range(1, n) : # rows
		for j in range(i) : # columns

			# Construct filehandle to be passed to riddersdiffcross
			fxixjh = lambda xixj : fxixj(f,xflat,i,j,xixj)

			# Calculate cross-derivative
			[hessf[i,j], err[i,j]] = tapas_riddersdiffcross(fxixjh,np.array([xflat[i],xflat[j]]),
					init_h     = init_h   ,
					div        = div      ,
					min_steps  = min_steps,
					max_steps  = max_steps,
					tf         = tf       )
			hessf[j,i] = hessf[i,j]
			err  [j,i] = err  [i,j]

	return [hessf, err]


def fxi(f,x,i,xi) :
	xx    = deepcopy(x)
	xx[i] = xi
	fxi   = f(xx)
	return fxi


def fxixj(f,x,i,j,xixj) :
	xx    = deepcopy(x)
	xx[i] = xixj[0]
	xx[j] = xixj[1]
	fxixj = f(xx)
	return fxixj
