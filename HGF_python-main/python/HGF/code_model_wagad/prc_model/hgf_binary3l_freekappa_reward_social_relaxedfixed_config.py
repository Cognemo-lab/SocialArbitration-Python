from .hgf_binary3l_freekappa_reward_social_fixed_config import (
    hgf_binary3l_freekappa_reward_social_fixed_config,
)


class hgf_binary3l_freekappa_reward_social_relaxedfixed_config(
    hgf_binary3l_freekappa_reward_social_fixed_config
):
    model = 'hgf_binary3l_freekappa_reward_social_relaxedfixed'
    fixed_parameters = ['mu2r_0', 'om_r', 'sa2a_0']
