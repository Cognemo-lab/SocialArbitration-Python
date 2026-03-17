import numpy as np


def linear_volatilitydecnoise_1stlevelprecision_reward_social_transp(r, ptrans):
    ptrans = np.asarray(ptrans)
    if ptrans.ndim == 1:
        ptrans = ptrans.reshape(1, -1)

    pvec = np.full_like(ptrans, np.nan, dtype=float)

    pvec[0, 0] = ptrans[0, 0]      # be0
    pvec[0, 1] = ptrans[0, 1]      # be1
    pvec[0, 2] = ptrans[0, 2]      # be2
    pvec[0, 3] = ptrans[0, 3]      # be3
    pvec[0, 4] = ptrans[0, 4]      # be4
    pvec[0, 5] = ptrans[0, 5]      # be5
    pvec[0, 6] = ptrans[0, 6]      # be6
    pvec[0, 7] = np.exp(ptrans[0, 7])  # ze
    pvec[0, 8] = np.exp(ptrans[0, 8])  # be_ch
    pvec[0, 9] = np.exp(ptrans[0, 9])  # be_wager

    pstruct = {
        'be0': pvec[0, 0],
        'be1': pvec[0, 1],
        'be2': pvec[0, 2],
        'be3': pvec[0, 3],
        'be4': pvec[0, 4],
        'be5': pvec[0, 5],
        'be6': pvec[0, 6],
        'ze': pvec[0, 7],
        'be_ch': pvec[0, 8],
        'be_wager': pvec[0, 9],
    }

    return [pvec, pstruct]
