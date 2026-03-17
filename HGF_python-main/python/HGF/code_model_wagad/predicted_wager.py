import numpy as np
from ..code_inversion.tapas_sgm import tapas_sgm


def calculate_predicted_wager(est):
    mu1hat_a = est['traj']['muhat_a'][:, 0]
    mu1hat_r = est['traj']['muhat_r'][:, 0]
    mu2hat_a = est['traj']['muhat_a'][:, 1]
    mu2hat_r = est['traj']['muhat_r'][:, 1]
    sa2hat_r = est['traj']['sahat_r'][:, 1]
    sa2hat_a = est['traj']['sahat_a'][:, 1]
    mu3hat_r = est['traj']['muhat_r'][:, 2]
    mu3hat_a = est['traj']['muhat_a'][:, 2]
    ze = est['p_obs']['ze']
    advice_card_space = est['u'][:, 2]

    transformed_mu1hat_r = mu1hat_r ** advice_card_space * (1.0 - mu1hat_r) ** (1.0 - advice_card_space)

    px = 1.0 / (mu1hat_a * (1.0 - mu1hat_a))
    pc = 1.0 / (mu1hat_r * (1.0 - mu1hat_r))
    wx = ze * px / (ze * px + pc)
    wc = pc / (ze * px + pc)
    b = wx * mu1hat_a + wc * transformed_mu1hat_r

    surp = -np.log2(b)
    arbitration = wx
    inferv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * sa2hat_a
    inferv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * sa2hat_r
    pv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * np.exp(mu3hat_a)
    pv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * np.exp(mu3hat_r)

    p_obs = est['p_obs']
    logrt = (
        p_obs['be0'] + p_obs['be1'] * surp + p_obs['be2'] * arbitration +
        p_obs['be3'] * inferv_a + p_obs['be4'] * inferv_r + p_obs['be5'] * pv_a +
        p_obs['be6'] * pv_r
    )

    sd = np.std(logrt, ddof=0)
    if sd == 0:
        return np.zeros_like(logrt)
    return (logrt - np.mean(logrt)) / sd
