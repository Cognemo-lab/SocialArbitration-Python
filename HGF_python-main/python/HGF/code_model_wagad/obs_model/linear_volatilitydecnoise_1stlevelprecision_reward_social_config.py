import numpy as np

from .linear_volatilitydecnoise_1stlevelprecision_reward_social import linear_volatilitydecnoise_1stlevelprecision_reward_social
from .linear_volatilitydecnoise_1stlevelprecision_reward_social_transp import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_transp,
)


class linear_volatilitydecnoise_1stlevelprecision_reward_social_config(object):
    """Config/priors for linear volatility-modulated decision-noise 1st-level
    precision reward-social observation model.

    Prior means and variances are empirical Bayes estimates derived from a
    prior sample, and therefore differ from the uninformative, zero-centred
    priors with wide variances used in the original MATLAB config
    (linear_1stlevelprecision_reward_social_config.m).  The MATLAB config
    uses be1mu–be6mu = 0, be0mu = log(500), logzemu = log(1),
    logbe_chmu = log(48), logbe_wagermu = log(5), and prior variances of 4
    (betas), 25 (zeta), 1 (be_ch), and 1000 (be_wager).
    """

    def __init__(self):
        self.model = 'linear_volatilitydecnoise_1stlevelprecision_reward_social'

        # Prior means (empirical Bayes estimates)
        self.be0mu = 6.2
        self.be1mu = -0.5
        self.be2mu = 0.5
        self.be3mu = -0.3
        self.be4mu = -0.2
        self.be5mu = -1.0
        self.be6mu = 0.05
        self.logzemu = np.log(2.0)
        self.logbe_chmu = np.log(6.0)
        self.logbe_wagermu = np.log(7.0)

        # Prior variances (empirical Bayes estimates; narrower than MATLAB defaults)
        self.be0sa = 1.0
        self.be1sa = 1.0
        self.be2sa = 1.0
        self.be3sa = 1.0
        self.be4sa = 1.0
        self.be5sa = 1.0
        self.be6sa = 1.0
        self.logzesa = 4.0
        self.logbe_chsa = 0.5
        self.logbe_wagersa = 4.0

        self.priormus = np.array([[
            self.be0mu,
            self.be1mu,
            self.be2mu,
            self.be3mu,
            self.be4mu,
            self.be5mu,
            self.be6mu,
            self.logzemu,
            self.logbe_chmu,
            self.logbe_wagermu,
        ]])

        self.priorsas = np.array([[
            self.be0sa,
            self.be1sa,
            self.be2sa,
            self.be3sa,
            self.be4sa,
            self.be5sa,
            self.be6sa,
            self.logzesa,
            self.logbe_chsa,
            self.logbe_wagersa,
        ]])

        # Consistent with existing TAPAS-style config objects
        self.obs_fun = linear_volatilitydecnoise_1stlevelprecision_reward_social
        self.transp_obs_fun = linear_volatilitydecnoise_1stlevelprecision_reward_social_transp
