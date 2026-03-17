import numpy as np
from ...code_inversion.tapas_sgm import tapas_sgm


def linear_volatilitydecnoise_1stlevelprecision_reward_social_sim(r, inf_states, p):
    """Simulate choices and wagers under the linear volatility-modulated
    decision-noise 1st-level precision model.

    Generates trial-wise responses consistent with the generative model
    assumed during inversion (see
    linear_volatilitydecnoise_1stlevelprecision_reward_social.py).

    Parameters are received in **native space** (i.e., already transformed
    via _transp), matching the convention of TAPAS sim functions.

    Responses
    ---------
    y_ch    – binary choice (0 or 1), sampled from a Bernoulli with
              volatility-modulated inverse temperature.
    y_wager – continuous wager, sampled from a Gaussian around the linear
              predictor.

    Returns
    -------
    y       : (n, 2) array – simulated responses [wager (continuous), choice (binary)]
    prob    : (n,)   array – choice probability on each trial
    """
    p = np.asarray(p).reshape(-1)
    be0, be1, be2, be3, be4, be5, be6, ze, be_ch, be_wager = p

    n = inf_states.shape[0]

    # Recode advice into reward-location coordinates (inline, matching main model)
    u = np.asarray(r['u']).astype(float)
    reward_location = u[:, 0]
    advice_helpfulness = u[:, 1]
    advice_in_loc_coords = 1.0 - np.abs(reward_location - advice_helpfulness)

    mu1hat_a = inf_states[:, 0, 2]
    mu1hat_r = inf_states[:, 0, 0]
    mu2hat_a = inf_states[:, 1, 2]
    mu2hat_r = inf_states[:, 1, 0]
    sa2hat_r = inf_states[:, 1, 1]
    sa2hat_a = inf_states[:, 1, 3]
    mu3hat_r = inf_states[:, 2, 0]
    mu3hat_a = inf_states[:, 2, 2]

    transformed_mu1hat_r = (mu1hat_r ** advice_in_loc_coords
                            * (1.0 - mu1hat_r) ** (1.0 - advice_in_loc_coords))

    # Precision 1st level (Fisher information)
    px = 1.0 / (mu1hat_a * (1.0 - mu1hat_a))
    pc = 1.0 / (mu1hat_r * (1.0 - mu1hat_r))

    # Precision-weighted arbitration
    wx = ze * px / (ze * px + pc)
    wc = pc / (ze * px + pc)

    # Integrated belief
    b = wx * mu1hat_a + wc * transformed_mu1hat_r
    b = np.clip(b, 1e-8, 1.0 - 1e-8)

    # ---- Dynamic (volatility-modulated) decision noise ----
    # Native-space be_ch is already exp-transformed, so divide directly.
    # MATLAB: decision_noise = exp(-mu3hat_r - mu3hat_a - log(be_ch))
    decision_noise = np.exp(-mu3hat_r - mu3hat_a) / be_ch

    # Surprise of the belief (matching estimation model)
    surp = -np.log2(b)

    arbitration = wx

    # Inferential variance (estimation uncertainty)
    inferv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * sa2hat_a
    inferv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * sa2hat_r

    # Phasic volatility (environmental uncertainty)
    pv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * np.exp(mu3hat_a)
    pv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * np.exp(mu3hat_r)

    # Predicted wager (continuous)
    wager_pred = (be0 + be1 * surp + be2 * arbitration
                  + be3 * inferv_a + be4 * inferv_r
                  + be5 * pv_a + be6 * pv_r)

    # Choice probability (power-of-sigmoid with dynamic noise)
    prob = b ** decision_noise / (b ** decision_noise + (1.0 - b) ** decision_noise)

    # Sample responses
    y_ch = (np.random.rand(n) < prob).astype(float)          # binary choice
    y_wager = wager_pred + np.sqrt(be_wager) * np.random.randn(n)  # continuous wager

    # Return in [wager, choice] column order, matching estimation model convention
    y = np.column_stack([y_wager, y_ch])

    return y, prob
