import numpy as np
import math
import sys
from copy import deepcopy
from ..code_model1.prc_model.MS9_dmpad_hgf_ar1_lvl3_config import MS9_dmpad_hgf_ar1_lvl3_config
from ..code_model1.obs_model.MS9_dmpad_constant_voltemp_exp_config import MS9_dmpad_constant_voltemp_exp_config
from .tapas_quasinewton_optim_config import tapas_quasinewton_optim_config
from .tapas_riddershessian import tapas_riddershessian
from .tapas_nearest_psd import tapas_nearest_psd
from .tapas_Cov2Corr import tapas_Cov2Corr


def tapas_fitModel(responses, inputs,
		c_prc = None,
		c_obs = None,
		c_opt = None
		) :
	'''
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	This is the main function for fitting the parameters of a combination of perceptual and
	observation models, given inputs and responses.

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	USAGE:
	    est = tapas_fitModel(responses, inputs)

	INPUT ARGUMENTS:
	    responses          Array of binary responses (column vector)
	    inputs             Array of inputs (column vector)

	                       Code irregular (missed, etc.) responses as NaN. Such responses will be
	                       ignored. However, the trial as such will not be ignored and filtering will
	                       take place based on the input.

	                       To ignore a trial, code the input as NaN. In this case, filtering is
	                       suspended for this trial and all representations (i.e., inferences on
	                       hidden states) will remain constant.

	                       Note that an input is often a composite event, for example a cue-stimulus
	                       contingency. If the agent you are modeling is lerning such contingencies,
	                       inputs have to be coded in contingency space (e.g., blue cue -> reward as
	                       well as green cue -> no reward is coded as 1 while blue cue -> no reward as
	                       well as green cue -> reward is coded as 0). The same applies to responses.

	                       If needed for a specific application, responses and inputs can be
	                       matrices with further columns. The coding of irregular and ignored
	                       trials described above then applies to their first column.

	OUTPUT:
	    est.u              Input to agent (i.e., the inputs array from the arguments)
	    est.y              Observed responses (i.e., the responses array from the arguments)
	    est.irr            Index numbers of irregular trials
	    est.ign            Index numbers of ignored trials
	    est.c_prc          Configuration settings for the chosen perceptual model
	                       (see the configuration file of that model for details)
	    est.c_obs          Configuration settings for the chosen observation model
	                       (see the configuration file of that model for details)
	    est.c_opt          Configuration settings for the chosen optimization algorithm
	                       (see the configuration file of that algorithm for details)
	    est.optim          A place where information on the optimization results is stored
	                       (e.g., measures of model quality like LME, AIC, BIC, and posterior
	                       parameter correlation)
	    est.p_prc          Maximum-a-posteriori estimates of perceptual parameters
	                       (see the configuration file of your chosen perceptual model for details)
	    est.p_obs          Maximum-a-posteriori estimates of observation parameters
	                       (see the configuration file of your chosen observation model for details)
	    est.traj:          Trajectories of the environmental states tracked by the perceptual model
	                       (see the configuration file of that model for details)

	CONFIGURATION:
	    In order to fit a model in this framework, you have to make three choices:

	    (1) a perceptual model,
	    (2) an observation model, and
	    (3) an optimization algorithm.

	    The perceptual model can for example be a Bayesian generative model of the states of an
	    agent's environment (like the Hierarchical Gaussian Filter (HGF)) or a reinforcement learning
	    algorithm (like Rescorla-Wagner (RW)). It describes the states or values that
	    probabilistically determine observed responses.

	    The observation model describes how the states or values of the perceptual model map onto
	    responses. Examples are the softmax decision rule or the closely related unit-square sigmoid
	    decision model.

	    The optimization algorithm is used to determine the maximum-a-posteriori (MAP) estimates of
	    the parameters of both the perceptual and decision models. Its objective function is the
	    unnormalized log-posterior of all perceptual and observation parameters, given the data and
	    the perceptual and observation models. This corresponds to the log-joint of data and
	    parameters, given the models.

	    Perceptual and observation models have to be chosen so that they are compatible, while the
	    choice of optimization algorithm is independent. To choose a particular combination, make
	    your changes to the configuration section of this file below. Compatibility information can
	    be found there.

	    Once you have made your choice, go to the relevant configuration files (e.g.,
	    tapas_hgf_binary_config.m for a choice of r.c_prc = tapas_hgf_binary_config), read the model- and
	    algorithm-specific information there, and configure accordingly.

	    The choices configured below can be overriden on the command line. Usage then is:

	    est = tapas_fitModel(responses, inputs, prc_model, obs_model, opt_algo)

	    where the last three arguments are strings containing the names of the corresponding
	    configuration files (without the extension .m).

	NEW DATASETS:
	    When analyzing a new dataset, take your inputs and use 'tapas_bayes_optimal_config' (or
	    'tapas_bayes_optimal_binary_config' for binary inputs) as your observation model. This determines
	    the Bayes optimal perceptual parameters (given your current priors, so choose them wide and
	    loose to let the inputs influence the result). You can then use the optimal parameters as your
	    new prior means for the perceptual parameters.

	PLOTTING OF RESULTS:
	    To plot the trajectories of the inferred perceptual states (as implied by the estimated
	    parameters), there is a function <modelname>_plotTraj(...) for each perceptual model. This
	    takes the structure returned by tapas_fitModel(...) as its only argument.

	    Additionally, the function tapas_fit_plotCorr(...) plots the posterior correlation of the
	    estimated parameters. It takes the structure returned by tapas_fitModel(...) as its only
	    argument. Note that this function only works if the optimization algorithm makes the
	    posterior correlation available in est.optim.Corr.

	EXAMPLE:
	    est = tapas_fitModel(responses, inputs)
	    tapas_hgf_binary_plotTraj(est)
	    tapas_fit_plotCorr(est)

	--------------------------------------------------------------------------------------------------
	Copyright (C) 2012-2015 Christoph Mathys, TNU, UZH & ETHZ

	This file is part of the HGF toolbox, which is released under the terms of the GNU General Public
	Licence (GPL), version 3. You can redistribute it and/or modify it under the terms of the GPL
	(either version 3 or, at your option, any later version). For further details, see the file
	COPYING or <http://www.gnu.org/licenses/>.

	Translated to Python by Lukas Baeni, HEJUBA GmbH, 2021
	'''

	# Store responses, inputs, and information about irregular trials in newly
	# initialized structure r
	r = dataPrep(responses, inputs)

	####################################################################################################
	####################################################################################################
	#
	# THE DEFAULTS DEFINED HERE WILL BE OVERWRITTEN BY ANY ARGUMENTS GIVEN WHEN CALLING tapas_fitModel.m

	# Default perceptual model
	# ~~~~~~~~~~~~~~~~~~~~~~~~
	if c_prc == None :
		r['c_prc'] = MS9_dmpad_hgf_ar1_lvl3_config()
	else :
		r['c_prc'] = c_prc()

	# Default observation model
	# ~~~~~~~~~~~~~~~~~~~~~~~~~
	if c_obs == None :
		r['c_obs'] = MS9_dmpad_constant_voltemp_exp_config()
	else :
		r['c_obs'] = c_obs()

	# Default optimization algorithm
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	if c_opt == None :
		r['c_opt'] = tapas_quasinewton_optim_config()
	else :
		r['c_opt'] = c_opt()

	# END OF CONFIGURATION
	#
	####################################################################################################
	####################################################################################################

	# Replace placeholders in parameter vectors with their calculated values
	r['c_prc'].priormus = np.where(r['c_prc'].priormus ==  99991,  r['plh']['p99991'], r['c_prc'].priormus)
	r['c_prc'].priorsas = np.where(r['c_prc'].priorsas ==  99991,  r['plh']['p99991'], r['c_prc'].priorsas)

	r['c_prc'].priormus = np.where(r['c_prc'].priormus ==  99992,  r['plh']['p99992'], r['c_prc'].priormus)
	r['c_prc'].priorsas = np.where(r['c_prc'].priorsas ==  99992,  r['plh']['p99992'], r['c_prc'].priorsas)

	r['c_prc'].priormus = np.where(r['c_prc'].priormus ==  99993,  r['plh']['p99993'], r['c_prc'].priormus)
	r['c_prc'].priorsas = np.where(r['c_prc'].priorsas ==  99993,  r['plh']['p99993'], r['c_prc'].priorsas)

	r['c_prc'].priormus = np.where(r['c_prc'].priormus == -99993, -r['plh']['p99993'], r['c_prc'].priormus)
	r['c_prc'].priorsas = np.where(r['c_prc'].priorsas == -99993, -r['plh']['p99993'], r['c_prc'].priorsas)

	r['c_prc'].priormus = np.where(r['c_prc'].priormus ==  99994,  r['plh']['p99994'], r['c_prc'].priormus)
	r['c_prc'].priorsas = np.where(r['c_prc'].priorsas ==  99994,  r['plh']['p99994'], r['c_prc'].priorsas)

	r.pop('plh')

	# Estimate mode of posterior parameter distribution (MAP estimate)
	r = optim(r, r['c_prc'].prc_fun, r['c_obs'].obs_fun, r['c_opt'].opt_algo)

	# Separate perceptual and observation parameters
	n_prcpars = r['c_prc'].priormus.shape[1]
	ptrans_prc = r['optim']['final'][0:1,0:n_prcpars]
	ptrans_obs = r['optim']['final'][0:1,n_prcpars:]

	# Transform MAP parameters back to their native space
	[dummy, r['p_prc']]   = r['c_prc'].transp_prc_fun(r, ptrans_prc)
	[dummy, r['p_obs']]   = r['c_obs'].transp_obs_fun(r, ptrans_obs)
	r['p_prc']['p']       = r['c_prc'].transp_prc_fun(r, ptrans_prc)
	r['p_obs']['p']       = r['c_obs'].transp_obs_fun(r, ptrans_obs)

	# Store transformed MAP parameters
	r['p_prc']['ptrans'] = ptrans_prc
	r['p_obs']['ptrans'] = ptrans_obs

	# Store representations at MAP estimate, response predictions, and residuals
	[r['traj'], infStates] = r['c_prc'].prc_fun(r, r['p_prc']['ptrans'], 'trans')

	try :
		# TODO: fix
		[dummy, r['optim']['yhat'], r['optim']['res']] = r['c_obs'].obs_fun(r, infStates, r['p_obs']['ptrans'])

		# Calculate autocorrelation of residuals
		res = r['optim']['res']
		res = np.where(np.isnan(res), 0., res) # Set residuals of irregular trials to zero
		r['optim']['resAC'] = tapas_autocorr(res) # TODO

	except :
		pass

	# Print results
	ftbrm = ['p', 'ptrans']
	dispprc = {key:r['p_prc'][key] for key in r['p_prc'] if key not in ftbrm}
	dispobs = {key:r['p_obs'][key] for key in r['p_obs'] if key not in ftbrm}

	print(' ')
	print('Results:')
	print(' ')
	print('Parameter estimates for the perceptual model:')
	print(dispprc)
	if len(dispobs.keys()) > 0 :
		print(' ')
		print('Parameter estimates for the observation model:')
		print(dispobs)
	print('Model quality:');
	print('    LME (more is better): %f' % r['optim']['LME'])
	print('    AIC (less is better): %f' % r['optim']['AIC'])
	print('    BIC (less is better): %f' % r['optim']['BIC'])
	print(' ')
	print('    AIC and BIC are approximations to -2*LME = %f.' % (-2.*r['optim']['LME']))
	print(' ')

	return r
# function tapas_fitModel

# --------------------------------------------------------------------------------------------------
def dataPrep(responses, inputs) :

	# Initialize data structure to be returned
	r = {}

	# Store responses and inputs
	r['y']  = responses
	r['u']  = inputs

	# Determine ignored trials
	ign = []
	for k in range(len(r['u'][:,0])) :
		if np.isnan(r['u'][k,0]) :
			ign.append(k)

	r['ign'] = ign

	if len(ign) == 0 :
		ignout = 'none'
	else :
		ignout = ', '.join(['%f' % entry for entry in ign])
	print('Ignored trials: %s' % ignout)

	# Determine irregular trials
	irr = []
	for k in range(len(r['y'][:,0])) :
		if np.isnan(r['y'][k,0]) :
			irr.append(k)

	# Make sure every ignored trial is also irregular
	irr = list(set(ign+irr))

	r['irr'] = irr

	if len(irr) == 0 :
		irrout = 'none'
	else :
		irrout = ', '.join(['%f' % entry for entry in irr])
	print('Irregular trials: %s' % irrout)

	# Calculate placeholder values for configuration files
	r['plh'] = {}

	# First input
	# Usually a good choice for the prior mean of mu_1
	r['plh']['p99991'] = r['u'][0,0];

	# Variance of first 20 inputs
	# Usually a good choice for the prior variance of mu_1
	if len(r['u']) > 20 :
		r['plh']['p99992'] = np.var(r['u'][:20,0])
	else :
		r['plh']['p99992'] = np.var(r['u'][:,0])

	# Log-variance of first 20 inputs
	# Usually a good choice for the prior means of log(sa_1) and alpha
	if len(r['u']) > 20 :
		r['plh']['p99993'] = math.log(np.var(r['u'][:20,0]))
	else :
		r['plh']['p99993'] = math.log(np.var(r['u'][:,0]))

	# Log-variance of first 20 inputs minus two
	# Usually a good choice for the prior mean of omega_1
	if len(r['u']) > 20 :
		r['plh']['p99994'] = math.log(np.var(r['u'][:20,0]))-2
	else :
		r['plh']['p99994'] = math.log(np.var(r['u'][:,0]))-2


	return r
# end of dataPrep


def optim(r, prc_fun, obs_fun, opt_algo) :

	# Determine indices of parameters to optimize (i.e., those that are not fixed or NaN)
	opt_idx = np.concatenate([r['c_prc'].priorsas, r['c_obs'].priorsas], axis = 1)
	opt_idx = np.where(np.isnan(opt_idx), 0., opt_idx)
	opt_idx = np.flatnonzero(opt_idx)

	# Number of perceptual and observation parameters
	n_prcpars = r['c_prc'].priormus.shape[1]
	n_obspars = r['c_obs'].priormus.shape[1]

	# Construct the objective function to be MINIMIZED:
	# The negative log-joint as a function of a single parameter vector
	nlj = lambda p : negLogJoint(r, prc_fun, obs_fun, p[0:1,0:n_prcpars], p[0:1,n_prcpars:n_prcpars+n_obspars])

	# Use means of priors as starting values for optimization for optimized parameters (and as values
	# for fixed parameters)
	init = np.concatenate([r['c_prc'].priormus, r['c_obs'].priormus], axis = 1)

	# Check whether priors are in a region where the objective function can be evaluated
	[dummy1, dummy2, rval, err] = nlj(init)
	if rval != 0 :
		sys.exit(err)

	# Do an optimization run
	optres = optimrun(nlj, init, opt_idx, opt_algo, r['c_opt'])

	# Record optimization results
	r['optim'] = {}
	r['optim']['init' ] = optres['init' ]
	r['optim']['final'] = optres['final']
	r['optim']['H'    ] = optres['H'    ]
	r['optim']['Sigma'] = optres['Sigma']
	r['optim']['Corr' ] = optres['Corr' ]
	r['optim']['negLl'] = optres['negLl']
	r['optim']['negLj'] = optres['negLj']
	r['optim']['LME'  ] = optres['LME'  ]
	r['optim']['accu' ] = optres['accu' ]
	r['optim']['comp' ] = optres['comp' ]

	# Do further optimization runs with random initialization
	if hasattr(r['c_opt'], 'nRandInit') and r['c_opt'].nRandInit > 0 :
		# Initialize the random number generator
		if hasattr(r['c_opt'], 'randSeed') :
			np.random.seed(r['c_opt'].randSeed)
		else :
			np.random.seed()

		for i in range(r['c_opt'].nRandInit) :
			# Use prior mean as starting value for random draw
			init = np.concatenate([r['c_prc'].priormus, r['c_obs'].priormus], axis = 1)

			# Get standard deviations of parameter priors
			priorsds = np.sqrt(np.concatenate([r['c_prc'].priorsas, r['c_obs'].priorsas], axis = 1))
			optsds = priorsds[0:1,opt_idx]

			# Add random values to prior means, drawn from Gaussian with prior sd
			init[0:1,opt_idx] = init[0:1,opt_idx] + np.random.randn(1,len(opt_idx))*optsds

			# Check whether initialization point is in a region where the objective
			# function can be evaluated
			[dummy1, dummy2, rval, err] = nlj(init)
			if rval != 0 :
				sys.exit(err)

			# Do an optimization run
			optres = optimrun(nlj, init, opt_idx, opt_algo, r['c_opt'])

			# Record optimization if the LME is better than the previous record
			if optres['LME'] > r['optim']['LME'] :
				r['optim']['init' ] = optres['init' ]
				r['optim']['final'] = optres['final']
				r['optim']['H'    ] = optres['H'    ]
				r['optim']['Sigma'] = optres['Sigma']
				r['optim']['Corr' ] = optres['Corr' ]
				r['optim']['negLl'] = optres['negLl']
				r['optim']['negLj'] = optres['negLj']
				r['optim']['LME'  ] = optres['LME'  ]
				r['optim']['accu' ] = optres['accu' ]
				r['optim']['comp' ] = optres['comp' ]

	# Calculate AIC and BIC
	d = len(opt_idx)
	if r['y'].shape[0] > 0 :
		ndp = np.count_nonzero(~np.isnan(r['y'][:,0]))
	else :
		ndp = np.count_nonzero(~np.isnan(r['u'][:,0]))
	r['optim']['AIC']  = 2.*r['optim']['negLl'] + 2.*d
	r['optim']['BIC']  = 2.*r['optim']['negLl'] + d*math.log(ndp)

	return r
# end of optim


def negLogJoint(r, prc_fun, obs_fun, ptrans_prc, ptrans_obs) :
	'''
	Returns the the negative log-joint density for perceptual and observation parameters
	'''

	# Calculate perceptual trajectories. The columns of the matrix infStates contain the trajectories of
	# the inferred states (according to the perceptual model) that the observation model bases its
	# predictions on.
	try :
		[dummy, infStates] = prc_fun(r, ptrans_prc, 'trans')
	except :
		negLogJoint = (2-2**(-52))*2**1023
		negLogLl    = (2-2**(-52))*2**1023
		# Signal that something has gone wrong
		rval = -1
		err  = []
		return [negLogJoint, negLogLl, rval, err]

	# Calculate the log-likelihood of observed responses given the perceptual trajectories,
	# under the observation model
	trialLogLls = obs_fun(r, infStates, ptrans_obs)
	if trialLogLls.shape[1] > 1 :
		sys.exit(1)
	logLl = np.nansum(trialLogLls, axis = 0)[0]
	negLogLl = -logLl

	# Calculate the log-prior of the perceptual parameters.
	# Only parameters that are neither NaN nor fixed (non-zero prior variance) are relevant.
	prc_idx = np.nan_to_num(r['c_prc'].priorsas)
	prc_idx = np.flatnonzero(prc_idx)

	logPrcPriors = -0.5*np.log(8.*np.arctan(1.)*r['c_prc'].priorsas[0:1,prc_idx]) - 0.5*(ptrans_prc[0:1,prc_idx] - r['c_prc'].priormus[0:1,prc_idx])**2/r['c_prc'].priorsas[0:1,prc_idx]
	logPrcPrior  = np.sum(logPrcPriors)

	# Calculate the log-prior of the observation parameters.
	# Only parameters that are neither NaN nor fixed (non-zero prior variance) are relevant.
	obs_idx = np.nan_to_num(r['c_obs'].priorsas)
	obs_idx = np.flatnonzero(obs_idx)

	logObsPriors = -0.5*np.log(8.*np.arctan(1.)*r['c_obs'].priorsas[0:1,obs_idx]) - 0.5*(ptrans_obs[0:1,obs_idx] - r['c_obs'].priormus[0:1,obs_idx])**2/r['c_obs'].priorsas[0:1,obs_idx]
	logObsPrior  = np.sum(logObsPriors)

	negLogJoint = -(logLl + logPrcPrior + logObsPrior)

	# Signal that all has gone right
	err = []
	rval = 0.

	return [negLogJoint, negLogLl, rval, err]
# end of negLogJoint


def optimrun(nlj, init, opt_idx, opt_algo, c_opt) :
	'''
	Does one run of the optimization algorithm and returns results
	'''

	# The objective function is now the negative log joint restricted
	# with respect to the parameters that are not optimized
	obj_fun = lambda p_opt : restrictfun(nlj, init, opt_idx, p_opt)

	# Optimize
	print(' ')
	print('Optimizing...')
	optres = opt_algo(obj_fun, init[0:1,opt_idx].T,
			verbose = c_opt.verbose,
			tolGrad = c_opt.tolGrad,
			tolArg  = c_opt.tolArg ,
			maxStep = c_opt.maxStep,
			maxIter = c_opt.maxIter,
			maxRegu = c_opt.maxRegu,
			maxRst  = c_opt.maxRst )

	# Record initialization point
	optres['init'] = init

	# Replace optimized values in init with arg min values
	final = deepcopy(init)
	final[0:1,opt_idx] = optres['argMin'].T
	optres['final'] = final

	# Get the negative log-joint and negative log-likelihood
	[negLj, negLl, _, _] = nlj(final)

	# Calculate the covariance matrix Sigma and the log-model evidence (as approximated
	# by the negative variational free energy under the Laplace assumption).
	print(' ')
	print('Calculating the log-model evidence (LME)...')
	d     = len(opt_idx)

	# Numerical computation of the Hessian of the negative log-joint at the MAP estimate
	[H, _] = tapas_riddershessian(obj_fun, optres['argMin'], init_h = 1, min_steps = 10)

	# Use the Hessian from the optimization, if available,
	# if the numerical Hessian is not positive definite
	if np.isinf(H).any() or np.isnan(H).any() or (np.linalg.eigvals(H) <= 0).any() :
		if 'T' in optres.keys() :
			# Hessian of the negative log-joint at the MAP estimate
			# (avoid asymmetry caused by rounding errors)
			H = np.linalg.inv(optres['T'])
			# Parameter covariance
			Sigma = optres['T']
			# Ensure H and Sigma are positive semi-definite
			H = tapas_nearest_psd(H)
			Sigma = tapas_nearest_psd(Sigma)
			# Parameter correlation
			Corr = tapas_Cov2Corr(Sigma)
			# Log-model evidence ~ negative variational free energy
			LME = -optres['valMin'] + 0.5*np.log(1./np.linalg.det(H)) + d/2.*np.log(2.*np.pi)
		else :
			print('Warning: Cannot calculate Sigma and LME because the Hessian is not positive definite.')
	else :
		# Calculate parameter covariance
		Sigma = np.linalg.inv(H)
		# Ensure H and Sigma are positive semi-definite
		H = tapas_nearest_psd(H)
		Sigma = tapas_nearest_psd(Sigma)
		# Parameter correlation
		Corr = tapas_Cov2Corr(Sigma)
		# Log-model evidence ~ negative variational free energy
		LME = -optres['valMin'] + 0.5*np.log(1./np.linalg.det(H)) + d/2.*np.log(2.*np.pi)

	# Record results
	optres['H'    ] = H
	optres['Sigma'] = Sigma
	optres['Corr' ] = Corr
	optres['negLl'] = negLl
	optres['negLj'] = negLj
	optres['LME'  ] = LME

	# Calculate accuracy and complexity (LME = accu - comp)
	optres['accu' ] = -negLl
	optres['comp' ] = optres['accu'] - LME

	return optres
# end of optimrun


def restrictfun(f, arg, free_idx, free_arg) :
	'''
	This is a helper function for the construction of file handles to
	restricted functions.

	It returns the value of a function restricted to subset of the
	arguments of the input function handle. The input handle takes
	*one* vector as its argument.

	INPUT:
	  f            The input function handle
	  arg          The argument vector for the input function containing the
	               fixed values of the restricted arguments (plus dummy values
	               for the free arguments)
	  free_idx     The index numbers of the arguments that are not restricted
	  free_arg     The values of the free arguments
	'''

	# Replace the dummy arguments in arg
	arg[0,free_idx] = free_arg

	# Evaluate
	[val, _, _, _] = f(arg)

	return val
# end of restrictfun
