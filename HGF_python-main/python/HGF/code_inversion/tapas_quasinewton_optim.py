import sys
import numpy as np
import math
from ..helper import iscolumn
from .tapas_riddersgradient import tapas_riddersgradient


def tapas_quasinewton_optim(f, init,
		verbose = False,
		tolGrad = 1e-3,
		tolArg  = 1e-3,
		maxStep = 2,
		maxIter = int(1e3),
		maxRegu = 4,
		maxRst  = 4
		) :
	'''
	This function implements the quasi-Newton minimization algorithm
	introduced by Broyden, Fletcher, Goldfarb, and Shanno (BFGS).

	INPUT:
	    f            Function handle of the function to be optimised
	                 f needs to accept a one-dimensional array as input
	                 variable.
	    init         The point at which to initialize the algorithm

	                 Optional settings structure that can contain the
	                 following fields:
	      tolGrad    Convergence tolerance in the gradient
	      tolArg     Convergence tolerance in the argument
	      maxIter    Maximum number of iterations
	      maxRegu    Maximum number of regularizations
	      verbose    Boolean flag to turn output on (true) or off (false)

	OUTPUT:
	    optim        Structure containing results in the following fields
	      valMin     The value of the function at its minimum
	      argMin     The argument of the function at its minimum
	      T          The inverse Hessian at the minimum calculated as a
	                 byproduct of optimization

	--------------------------------------------------------------------------------------------------
	Copyright (C) 2012-2019 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	# Dimension count
	n = len(init)

	# Make sure init is a column vector
	if not iscolumn(init) :
		init = init.T
		if not iscolumn(init) :
			sys.exit('tapas:hgf:QuasinewtonOptim:InitPointNotRow   Initial point has to be a row vector.')

	# Evaluate initial value of objective function
	x = init
	val = f(x.flatten())

	if verbose :
		print(' ')
		print('Initial argument: %s' % np.array2string(x.T))
		print('Initial value   : %f' % val                 )

	# Calculate gradient
	gradoptions = {}
	gradoptions['min_steps'] = 10
	[grad, err] = tapas_riddersgradient(f, x, **gradoptions)

	# Initialize negative Sigma (here called T) as the unit matrix
	T = np.eye(n)

	# Initialize descent vector and slope
	descvec = -grad.T
	slope   = np.matmul(grad,descvec)

	# Initialize new point and new value
	newx    = np.nan
	newval  = np.nan
	dval    = np.nan

	# Initialize reset count
	resetcount = 0

	# Iterate
	for i in range(maxIter) :

		if verbose :
			print('\n==================================')
			print('Iteration %d\n' % i)

		# Limit step size
		stepSize = math.sqrt(np.matmul(descvec.T,descvec))
		if stepSize > maxStep :
			descvec = descvec*maxStep/math.sqrt(np.matmul(descvec.T,descvec))

		regucount = 0
		# Move in the descent direction, looping through regularizations
		for j in range(maxRegu+1) :
			regucount = j

			# Begin with t=1 and halve on each step
			t       = 0.5**j
			newx    = x+t*descvec
			newval  = f(newx.flatten())

			# Regularize if the objective function value is Inf
			if np.isinf(newval) :
				continue
			else :
				dval = newval-val

			# Stop if the new value is sufficiently smaller
			if dval < 1e-4*t*slope :
				break

		# Update point and value if regularizations have not been exhausted;
		# otherwise, reset and start again by jumping back 10% of the way to
		# the first initialization.
		if regucount < maxRegu :
			dx   = newx-x
			x    = newx
			val  = newval
		elif resetcount < maxRst :
			T       = np.eye(n)
			x       = x+0.1*(init-x)
			val     = f(x.flatten())

			[grad, err] = tapas_riddersgradient(f, x, **gradoptions)
			descvec = -grad.T
			slope   = np.matmul(grad,descvec)

			i = -1
			resetcount = resetcount+1

			if  verbose :
				print(' ')
				print('Regularizations exhausted - resetting algorithm.')
				print('Initial argument : %s' % np.array2string(x.T, prefix = 'Initial argument : '))
				print('Initial value    : %f' % val)
			continue
		else :
			print(' ')
			print('Warning: optimization terminated because the maximum number of resets was reached.')
			break

		if verbose :
			print(' ')
			print('Argument        : %s' % np.array2string(x.T, prefix = 'Argument        : '))
			print('Value           : %f' % val                                                )
			print('Improvement     : %f' % -dval                                              )
			print('Regularizations : %d' % regucount                                          )

		# Test for convergence
		if np.max(abs(dx)/abs(np.maximum(x,1))) < tolArg :
			if verbose :
				print(' ')
				print('Converged on step size')
			break

		# Update gradient
		oldgrad = grad
		[grad, err] = tapas_riddersgradient(f, x, **gradoptions)
		dgrad   = grad-oldgrad

		# Test for convergence
		if np.max(abs(grad.T)*np.maximum(abs(x),1.)/max(abs(val),1.)) < tolGrad :
			if verbose :
				print(' ')
				print('Converged on gradient size')
			break

		# Update T according to BFGS
		if np.matmul(dgrad,dx) > math.sqrt(np.finfo(float).eps*(np.matmul(dgrad,dgrad.T))*np.matmul(dx.T,dx)) :

			dgdx  = np.matmul(dgrad,dx)
			dgT   = np.matmul(dgrad,T)
			dgTdg = np.matmul(dgrad,np.matmul(T,dgrad.T))
			u     = dx/dgdx-dgT.T/dgTdg

			T = T + np.matmul(dx,dx.T)/dgdx - np.matmul(dgT.T,dgT)/dgTdg + dgTdg*np.matmul(u,u.T)

		# Update descent vector
		descvec = -np.matmul(T,grad.T)

		# Update slope
		slope = np.matmul(grad,descvec)

		# Warn if termination is only due to maximum of iterations being reached
		if i == maxIter - 1 :
			print(' ')
			print('Warning: optimization terminated because the maximum number of iterations was reached.')

	# Collect results
	optim = {}
	optim['valMin'] = val
	optim['argMin'] = x
	optim['T'     ] = T

	return optim
