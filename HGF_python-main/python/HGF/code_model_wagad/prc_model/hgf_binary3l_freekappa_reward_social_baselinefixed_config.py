from .hgf_binary3l_freekappa_reward_social_fixed_config import (
    hgf_binary3l_freekappa_reward_social_fixed_config,
)


class hgf_binary3l_freekappa_reward_social_baselinefixed_config(
    hgf_binary3l_freekappa_reward_social_fixed_config
):
    model = 'hgf_binary3l_freekappa_reward_social_baselinefixed'
    fixed_parameters = ['mu2r_0', 'mu2a_0', 'mu3a_0', 'om_r', 'sa2a_0']
