import numpy as np
from ...code_inversion.tapas_logit import tapas_logit
from .hgf_binary3l_freekappa_reward_social import hgf_binary3l_freekappa_reward_social
from .hgf_binary3l_freekappa_reward_social_transp import hgf_binary3l_freekappa_reward_social_transp


class hgf_binary3l_freekappa_reward_social_config(object):
    """Config/priors for the 3-level binary HGF with parallel reward and
    social (advice) learning branches, with free kappa.

    Fixed vs free parameters
    ------------------------
    The following parameters are FIXED (prior variance = 0):
      phi_r    – mean-reversion rate reward (fixed at 0.1)
      phi_a    – mean-reversion rate advice (fixed at 0.1)

    The following parameters are FREE (prior variance > 0):
      mu2r_0, sa2r_0, mu3r_0, sa3r_0, ka_r, om_r, th_r  (reward branch)
      mu2a_0, sa2a_0, mu3a_0, sa3a_0, ka_a, om_a, th_a   (advice branch)
      m_r, m_a                                              (mean-reversion targets)

    Note: the MATLAB config (hgf_binary3l_reward_social_config.m) differs —
    it fixes ka_r, ka_a, phi_r, phi_a and uses the same initial-value
    free/fixed pattern for mu/sa _0 parameters.  The present Python config
    reflects the intended model specification with free kappa and all
    initial values free.

    Parameter ordering (indices 15–18) uses the interleaved convention
    [phi_r, phi_a, m_r, m_a] in the prior vectors, whereas the MATLAB
    config groups them identically but the _transp files differ: Python
    _transp interleaves [phi_r, m_r, phi_a, m_a].
    """

    def __init__(self):
        self.model = 'hgf_binary3l_freekappa_reward_social'
        self.irregular_intervals = False

        self.kaub_r = 1.0
        self.thub_r = 1.0
        self.kaub_a = 1.0
        self.thub_a = 1.0

        self.mu2r_0mu, self.mu2r_0sa = 0.0, 1.0          # free
        self.logsa2r_0mu, self.logsa2r_0sa = np.log(1.0), 1.0  # free
        self.mu3r_0mu, self.mu3r_0sa = 1.0, 1.0          # free
        self.logsa3r_0mu, self.logsa3r_0sa = np.log(1.0), 1.0  # free
        self.logitkamu_r, self.logitkasa_r = 0.0, 1.0    # free
        self.ommu_r, self.omsa_r = -4.0, 16.0             # free
        self.logitthmu_r, self.logitthsa_r = 0.25, 16.0   # free

        self.mu2a_0mu, self.mu2a_0sa = 0.0, 1.0          # free
        self.logsa2a_0mu, self.logsa2a_0sa = np.log(1.0), 1.0  # free
        self.mu3a_0mu, self.mu3a_0sa = 1.0, 1.0          # free
        self.logsa3a_0mu, self.logsa3a_0sa = np.log(1.0), 1.0  # free
        self.logitkamu_a, self.logitkasa_a = 0.0, 1.0    # free
        self.ommu_a, self.omsa_a = -4.0, 16.0             # free
        self.logitthmu_a, self.logitthsa_a = 0.25, 16.0   # free

        self.logitphimu_r, self.logitphisa_r = tapas_logit(0.1, 1.0), 0.0  # fixed
        self.logitphimu_a, self.logitphisa_a = tapas_logit(0.1, 1.0), 0.0  # fixed
        self.mmu_r, self.msa_r = self.mu3r_0mu, 1.0       # free
        self.mmu_a, self.msa_a = self.mu3a_0mu, 1.0       # free

        self.priormus = np.array([[
            self.mu2r_0mu, self.logsa2r_0mu, self.mu3r_0mu, self.logsa3r_0mu,
            self.logitkamu_r, self.ommu_r, self.logitthmu_r,
            self.mu2a_0mu, self.logsa2a_0mu, self.mu3a_0mu, self.logsa3a_0mu,
            self.logitkamu_a, self.ommu_a, self.logitthmu_a,
            self.logitphimu_r, self.logitphimu_a, self.mmu_r, self.mmu_a
        ]])

        self.priorsas = np.array([[
            self.mu2r_0sa, self.logsa2r_0sa, self.mu3r_0sa, self.logsa3r_0sa,
            self.logitkasa_r, self.omsa_r, self.logitthsa_r,
            self.mu2a_0sa, self.logsa2a_0sa, self.mu3a_0sa, self.logsa3a_0sa,
            self.logitkasa_a, self.omsa_a, self.logitthsa_a,
            self.logitphisa_r, self.logitphisa_a, self.msa_r, self.msa_a
        ]])

        self.prc_fun = hgf_binary3l_freekappa_reward_social
        self.transp_prc_fun = hgf_binary3l_freekappa_reward_social_transp
