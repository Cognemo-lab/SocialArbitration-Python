import numpy as np
from ...code_inversion.tapas_sgm import tapas_sgm


def hgf_binary3l_freekappa_reward_social_transp(r, ptrans):
    ptrans = np.asarray(ptrans).reshape(-1)
    pvec = np.full((18,), np.nan, dtype=float)

    pvec[0] = ptrans[0]
    pvec[1] = np.exp(ptrans[1])
    pvec[2] = ptrans[2]
    pvec[3] = np.exp(ptrans[3])
    pvec[4] = tapas_sgm(ptrans[4], r['c_prc'].kaub_r)
    pvec[5] = ptrans[5]
    pvec[6] = tapas_sgm(ptrans[6], r['c_prc'].thub_r)

    pvec[7] = ptrans[7]
    pvec[8] = np.exp(ptrans[8])
    pvec[9] = ptrans[9]
    pvec[10] = np.exp(ptrans[10])
    pvec[11] = tapas_sgm(ptrans[11], r['c_prc'].kaub_a)
    pvec[12] = ptrans[12]
    pvec[13] = tapas_sgm(ptrans[13], r['c_prc'].thub_a)

    pvec[14] = tapas_sgm(ptrans[14], 1.0)
    pvec[15] = ptrans[15]
    pvec[16] = tapas_sgm(ptrans[16], 1.0)
    pvec[17] = ptrans[17]

    pstruct = {
        'mu2r_0': pvec[0], 'sa2r_0': pvec[1], 'mu3r_0': pvec[2], 'sa3r_0': pvec[3],
        'ka_r': pvec[4], 'om_r': pvec[5], 'th_r': pvec[6],
        'mu2a_0': pvec[7], 'sa2a_0': pvec[8], 'mu3a_0': pvec[9], 'sa3a_0': pvec[10],
        'ka_a': pvec[11], 'om_a': pvec[12], 'th_a': pvec[13],
        'phi_r': pvec[14], 'm_r': pvec[15], 'phi_a': pvec[16], 'm_a': pvec[17],
    }

    return [pvec.reshape(1, -1), pstruct]
