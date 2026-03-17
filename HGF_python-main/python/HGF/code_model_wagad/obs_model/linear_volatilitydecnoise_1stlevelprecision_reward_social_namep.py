import numpy as np


def linear_volatilitydecnoise_1stlevelprecision_reward_social_namep(pvec):
    p = np.asarray(pvec).reshape(-1)
    return {
        'be0': p[0], 'be1': p[1], 'be2': p[2], 'be3': p[3], 'be4': p[4],
        'be5': p[5], 'be6': p[6], 'ze': p[7], 'be_ch': p[8], 'be_wager': p[9],
    }
