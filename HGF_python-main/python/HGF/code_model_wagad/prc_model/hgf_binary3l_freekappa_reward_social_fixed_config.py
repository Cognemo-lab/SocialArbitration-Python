from .hgf_binary3l_freekappa_reward_social_config import hgf_binary3l_freekappa_reward_social_config


class hgf_binary3l_freekappa_reward_social_fixed_config(hgf_binary3l_freekappa_reward_social_config):
    """Variant of the reward-social HGF config that can fix named parameters.

    Parameters are fixed by setting their prior variance to zero. If a fixed
    value is provided in ``fixed_parameters`` it also overwrites the prior mean.
    Otherwise the inherited prior mean is used as the fixed value.
    """

    model = 'hgf_binary3l_freekappa_reward_social_fixed'

    # Either:
    #   {'ka_r': 0.5, 'm_a': 1.0}
    # or:
    #   ['ka_r', 'm_a']
    fixed_parameters = {}

    _PARAM_TO_PRIOR_INDEX = {
        'mu2r_0': 0,
        'sa2r_0': 1,
        'mu3r_0': 2,
        'sa3r_0': 3,
        'ka_r': 4,
        'om_r': 5,
        'th_r': 6,
        'mu2a_0': 7,
        'sa2a_0': 8,
        'mu3a_0': 9,
        'sa3a_0': 10,
        'ka_a': 11,
        'om_a': 12,
        'th_a': 13,
        'phi_r': 14,
        'phi_a': 15,
        'm_r': 16,
        'm_a': 17,
    }

    def __init__(self):
        super().__init__()
        self.model = self.__class__.model
        self._apply_fixed_parameters()

    def _apply_fixed_parameters(self):
        fixed = getattr(self, 'fixed_parameters', {})
        if isinstance(fixed, (list, tuple, set)):
            fixed = {name: None for name in fixed}

        for name, value in fixed.items():
            if name not in self._PARAM_TO_PRIOR_INDEX:
                raise ValueError(f'Unknown perceptual parameter to fix: {name}')
            idx = self._PARAM_TO_PRIOR_INDEX[name]
            if value is not None:
                self.priormus[0, idx] = value
            self.priorsas[0, idx] = 0.0
