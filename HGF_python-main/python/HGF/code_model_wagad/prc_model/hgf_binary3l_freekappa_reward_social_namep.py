import numpy as np


def hgf_binary3l_freekappa_reward_social_namep(pvec):
    p = np.asarray(pvec).reshape(-1)
    return {
        'mu2r_0': p[0], 'sa2r_0': p[1], 'mu3r_0': p[2], 'sa3r_0': p[3],
        'ka_r': p[4], 'om_r': p[5], 'th_r': p[6],
        'mu2a_0': p[7], 'sa2a_0': p[8], 'mu3a_0': p[9], 'sa3a_0': p[10],
        'ka_a': p[11], 'om_a': p[12], 'th_a': p[13],
        'phi_r': p[14], 'm_r': p[15], 'phi_a': p[16], 'm_a': p[17],
    }
