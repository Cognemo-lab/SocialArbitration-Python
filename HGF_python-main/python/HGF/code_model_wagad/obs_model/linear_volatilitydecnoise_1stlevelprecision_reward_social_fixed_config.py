from .linear_volatilitydecnoise_1stlevelprecision_reward_social_config import (
    linear_volatilitydecnoise_1stlevelprecision_reward_social_config,
)


class linear_volatilitydecnoise_1stlevelprecision_reward_social_fixed_config(
    linear_volatilitydecnoise_1stlevelprecision_reward_social_config
):
    """Variant of the observation config that can fix named parameters."""

    model = 'linear_volatilitydecnoise_1stlevelprecision_reward_social_fixed'

    # Either:
    #   {'be1': 0.0, 'be_ch': 2.0}
    # or:
    #   ['be1', 'be_ch']
    fixed_parameters = {}

    _PARAM_TO_PRIOR_INDEX = {
        'be0': 0,
        'be1': 1,
        'be2': 2,
        'be3': 3,
        'be4': 4,
        'be5': 5,
        'be6': 6,
        'ze': 7,
        'be_ch': 8,
        'be_wager': 9,
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
                raise ValueError(f'Unknown observation parameter to fix: {name}')
            idx = self._PARAM_TO_PRIOR_INDEX[name]
            if value is not None:
                self.priormus[0, idx] = value
            self.priorsas[0, idx] = 0.0
