import numpy as np
from ...code_inversion.tapas_sgm import tapas_sgm


def linear_volatilitydecnoise_1stlevelprecision_reward_social(r, inf_states, ptrans):
    """Log-likelihood for choices+wagers under linear 1st-level precision model.

    Uses dynamic (volatility-modulated) inverse temperature for the choice
    likelihood, consistent with the MATLAB reference implementation.

    Responses
    ---------
    The model jointly fits two behavioural outcomes per trial:
      y_ch    – binary choice (0 or 1), modelled with a power-of-sigmoid
                (Bernoulli) likelihood parameterised by the integrated belief
                and a volatility-modulated inverse temperature.
      y_wager – continuous wager amount, modelled with a Gaussian likelihood
                around a linear combination of computational quantities
                (surprise, arbitration, inferential variance, phasic
                volatility).

    Returns
    -------
    logp  : (n, 1) array – trial-wise log-probability (NaN for irregular trials)
    yhat  : (n, 1) array – predicted wager value (continuous)
    res   : (n, 1) array – wager residual (observed − predicted)
    """

    ptrans = np.asarray(ptrans)
    if ptrans.ndim == 2:
        ptrans = ptrans.reshape(-1)

    be0 = ptrans[0]
    be1 = ptrans[1]
    be2 = ptrans[2]
    be3 = ptrans[3]
    be4 = ptrans[4]
    be5 = ptrans[5]
    be6 = ptrans[6]
    ze = np.exp(ptrans[7])
    be_ch = np.exp(ptrans[8])
    be_wager = np.exp(ptrans[9])

    n = inf_states.shape[0]
    logp = np.full((n, 1), np.nan, dtype=float)
    yhat = np.full((n, 1), np.nan, dtype=float)
    res = np.full((n, 1), np.nan, dtype=float)

    irr = set(r.get('irr', []))

    # NOTE: Column order follows MATLAB convention:
    #   y[:, 0] = wager (continuous),  y[:, 1] = choice (binary)
    # Verify that your data loader populates columns accordingly.
    y_ch = np.asarray(r['y'])[:, 1].astype(float)       # binary choice (0 or 1)
    y_wager = np.asarray(r['y'])[:, 0].astype(float)     # continuous wager amount

    # Recode advice into reward-location coordinates
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # MATLAB: cardColor = r.u(:,1); adviceHelpfulness = r.u(:,2);
    #         advice_card_space = recodeAdviceToColor(cardColor, adviceHelpfulness);
    #
    # Logic (XNOR): when advice is helpful it points to the reward location;
    # when misleading it points to the opposite location.
    #   helpful & reward=1  (1,1) → 1
    #   helpful & reward=0  (1,0) → 0
    #   misleading & reward=1  (0,1) → 0
    #   misleading & reward=0  (0,0) → 1
    #
    # TODO: verify that this matches your recodeAdviceToColor.m exactly.
    u = np.asarray(r['u']).astype(float)
    reward_location = u[:, 0]       # MATLAB r.u(:,1) – location of reward
    advice_helpfulness = u[:, 1]    # MATLAB r.u(:,2)
    advice_in_loc_coords = 1.0 - np.abs(reward_location - advice_helpfulness)

    mu1hat_a = inf_states[:, 0, 2]
    mu1hat_r = inf_states[:, 0, 0]
    mu2hat_a = inf_states[:, 1, 2]
    mu2hat_r = inf_states[:, 1, 0]
    sa2hat_r = inf_states[:, 1, 1]
    sa2hat_a = inf_states[:, 1, 3]
    mu3hat_r = inf_states[:, 2, 0]
    mu3hat_a = inf_states[:, 2, 2]

    transformed_mu1hat_r = mu1hat_r ** advice_in_loc_coords * (1.0 - mu1hat_r) ** (1.0 - advice_in_loc_coords)

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
    # MATLAB: decision_noise = exp((-mu3hat_r) + (-mu3hat_a) - log(be_ch))
    #       = exp(-mu3hat_r - mu3hat_a) / be_ch
    decision_noise = np.exp(-mu3hat_r - mu3hat_a) / be_ch

    surp = -np.log2(b)
    arbitration = wx

    # Inferential variance (estimation uncertainty)
    inferv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * sa2hat_a
    inferv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * sa2hat_r

    # Phasic volatility (environmental uncertainty)
    pv_a = tapas_sgm(mu2hat_a, 1.0) * (1.0 - tapas_sgm(mu2hat_a, 1.0)) * np.exp(mu3hat_a)
    pv_r = tapas_sgm(mu2hat_r, 1.0) * (1.0 - tapas_sgm(mu2hat_r, 1.0)) * np.exp(mu3hat_r)

    # Predicted wager
    wager_pred = (be0 + be1 * surp + be2 * arbitration
                  + be3 * inferv_a + be4 * inferv_r
                  + be5 * pv_a + be6 * pv_r)

    # Choice log-likelihood: Bernoulli with power-of-sigmoid link and
    # dynamic (volatility-modulated) inverse temperature (binary y_ch ∈ {0,1})
    logp_ch = (y_ch * decision_noise * np.log(b / (1.0 - b))
               + np.log(((1.0 - b) ** decision_noise)
                        / (((1.0 - b) ** decision_noise) + (b ** decision_noise))))

    # Wager log-likelihood: Gaussian around linear predictor (continuous y_wager)
    logp_wager = (-0.5 * np.log(2.0 * np.pi * be_wager)
                  - ((y_wager - wager_pred) ** 2) / (2.0 * be_wager))

    # Mask irregular and non-finite trials
    valid = np.ones(n, dtype=bool)
    if len(irr) > 0:
        valid[list(irr)] = False
    valid = valid & np.isfinite(y_ch) & np.isfinite(y_wager)

    logp[valid, 0] = logp_ch[valid] + logp_wager[valid]
    yhat[valid, 0] = wager_pred[valid]
    res[valid, 0] = y_wager[valid] - wager_pred[valid]

    return logp, yhat, res
