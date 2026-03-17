import numpy as np
from .hgf_binary3l_freekappa_reward_social_transp import hgf_binary3l_freekappa_reward_social_transp
from ...code_inversion.tapas_sgm import tapas_sgm


def hgf_binary3l_freekappa_reward_social(r, p, varargin=None):
    if varargin is not None and varargin == 'trans':
        p, _ = hgf_binary3l_freekappa_reward_social_transp(r, p)

    p = np.asarray(p).reshape(-1)

    mu2r_0, sa2r_0, mu3r_0, sa3r_0, ka_r, om_r, th_r = p[0:7]
    mu2a_0, sa2a_0, mu3a_0, sa3a_0, ka_a, om_a, th_a = p[7:14]
    phi_r, m_r, phi_a, m_a = p[14:18]

    u_r = np.concatenate(([0.0], np.asarray(r['u'])[:, 1]))
    u_a = np.concatenate(([0.0], np.asarray(r['u'])[:, 0]))
    n = len(u_r)
    t_r = np.ones(n)
    t_a = np.ones(n)

    mu1_r = np.full(n, np.nan)
    mu2_r = np.full(n, np.nan)
    pi2_r = np.full(n, np.nan)
    mu3_r = np.full(n, np.nan)
    pi3_r = np.full(n, np.nan)
    mu1_a = np.full(n, np.nan)
    mu2_a = np.full(n, np.nan)
    pi2_a = np.full(n, np.nan)
    mu3_a = np.full(n, np.nan)
    pi3_a = np.full(n, np.nan)

    mu1hat_r = np.full(n, np.nan)
    pi1hat_r = np.full(n, np.nan)
    pi2hat_r = np.full(n, np.nan)
    pi3hat_r = np.full(n, np.nan)
    w2_r = np.full(n, np.nan)
    da1_r = np.full(n, np.nan)
    da2_r = np.full(n, np.nan)

    mu1hat_a = np.full(n, np.nan)
    pi1hat_a = np.full(n, np.nan)
    pi2hat_a = np.full(n, np.nan)
    pi3hat_a = np.full(n, np.nan)
    w2_a = np.full(n, np.nan)
    da1_a = np.full(n, np.nan)
    da2_a = np.full(n, np.nan)

    mu1_r[0] = tapas_sgm(mu2r_0, 1.0)
    mu2_r[0] = mu2r_0
    pi2_r[0] = 1.0 / sa2r_0
    mu3_r[0] = mu3r_0
    pi3_r[0] = 1.0 / sa3r_0

    mu1_a[0] = tapas_sgm(mu2a_0, 1.0)
    mu2_a[0] = mu2a_0
    pi2_a[0] = 1.0 / sa2a_0
    mu3_a[0] = mu3a_0
    pi3_a[0] = 1.0 / sa3a_0

    ign = set(np.asarray(r.get('ign', []), dtype=int).tolist())

    for k in range(1, n):
        if (k - 1) not in ign:
            mu1hat_r[k] = tapas_sgm(mu2_r[k - 1], 1.0)
            mu1hat_a[k] = tapas_sgm(mu2_a[k - 1], 1.0)
            pi1hat_r[k] = 1.0 / (mu1hat_r[k] * (1.0 - mu1hat_r[k]))
            pi1hat_a[k] = 1.0 / (mu1hat_a[k] * (1.0 - mu1hat_a[k]))

            mu1_r[k] = u_r[k]
            mu1_a[k] = u_a[k]
            da1_r[k] = mu1_r[k] - mu1hat_r[k]
            da1_a[k] = mu1_a[k] - mu1hat_a[k]

            pi2hat_r[k] = 1.0 / (1.0 / pi2_r[k - 1] + t_r[k] * np.exp(ka_r * mu3_r[k - 1] + om_r))
            pi2hat_a[k] = 1.0 / (1.0 / pi2_a[k - 1] + t_a[k] * np.exp(ka_a * mu3_a[k - 1] + om_a))
            pi2_r[k] = pi2hat_r[k] + 1.0 / pi1hat_r[k]
            pi2_a[k] = pi2hat_a[k] + 1.0 / pi1hat_a[k]

            mu2_r[k] = mu2_r[k - 1] + da1_r[k] / pi2_r[k]
            mu2_a[k] = mu2_a[k - 1] + da1_a[k] / pi2_a[k]

            da2_r[k] = (1.0 / pi2_r[k] + (mu2_r[k] - mu2_r[k - 1]) ** 2) * pi2hat_r[k] - 1.0
            da2_a[k] = (1.0 / pi2_a[k] + (mu2_a[k] - mu2_a[k - 1]) ** 2) * pi2hat_a[k] - 1.0

            pi3hat_r[k] = 1.0 / (1.0 / pi3_r[k - 1] + t_r[k] * th_r)
            pi3hat_a[k] = 1.0 / (1.0 / pi3_a[k - 1] + t_a[k] * th_a)

            w2_r[k] = t_r[k] * np.exp(ka_r * mu3_r[k - 1] + om_r) * pi2hat_r[k]
            w2_a[k] = t_a[k] * np.exp(ka_a * mu3_a[k - 1] + om_a) * pi2hat_a[k]

            pi3_r[k] = pi3hat_r[k] + 0.5 * ka_r ** 2 * w2_r[k] * (w2_r[k] + (2.0 * w2_r[k] - 1.0) * da2_r[k])
            pi3_a[k] = pi3hat_a[k] + 0.5 * ka_a ** 2 * w2_a[k] * (w2_a[k] + (2.0 * w2_a[k] - 1.0) * da2_a[k])

            if pi3_r[k] <= 0 or pi3_a[k] <= 0:
                raise ValueError('Negative posterior precision: parameters violate model assumptions.')

            mu3_r[k] = mu3_r[k - 1] + 0.5 * ka_r * w2_r[k] * da2_r[k] / pi3_r[k]
            mu3_a[k] = mu3_a[k - 1] + 0.5 * ka_a * w2_a[k] * da2_a[k] / pi3_a[k]
        else:
            mu1_r[k], mu2_r[k], pi2_r[k], mu3_r[k], pi3_r[k] = mu1_r[k - 1], mu2_r[k - 1], pi2_r[k - 1], mu3_r[k - 1], pi3_r[k - 1]
            mu1_a[k], mu2_a[k], pi2_a[k], mu3_a[k], pi3_a[k] = mu1_a[k - 1], mu2_a[k - 1], pi2_a[k - 1], mu3_a[k - 1], pi3_a[k - 1]
            mu1hat_r[k], pi1hat_r[k], pi2hat_r[k], pi3hat_r[k], w2_r[k], da1_r[k], da2_r[k] = mu1hat_r[k - 1], pi1hat_r[k - 1], pi2hat_r[k - 1], pi3hat_r[k - 1], w2_r[k - 1], da1_r[k - 1], da2_r[k - 1]
            mu1hat_a[k], pi1hat_a[k], pi2hat_a[k], pi3hat_a[k], w2_a[k], da1_a[k], da2_a[k] = mu1hat_a[k - 1], pi1hat_a[k - 1], pi2hat_a[k - 1], pi3hat_a[k - 1], w2_a[k - 1], da1_a[k - 1], da2_a[k - 1]

    mu2hat_r = mu2_r[:-1]
    mu3hat_r = (mu3_r + phi_r * (m_r - mu3_r))[:-1]
    mu2hat_a = mu2_a[:-1]
    mu3hat_a = (mu3_a + phi_a * (m_a - mu3_a))[:-1]

    mu1_r = mu1_r[1:]
    mu2_r = mu2_r[1:]
    pi2_r = pi2_r[1:]
    mu3_r = mu3_r[1:]
    pi3_r = pi3_r[1:]
    mu1_a = mu1_a[1:]
    mu2_a = mu2_a[1:]
    pi2_a = pi2_a[1:]
    mu3_a = mu3_a[1:]
    pi3_a = pi3_a[1:]

    mu1hat_r = mu1hat_r[1:]
    pi1hat_r = pi1hat_r[1:]
    pi2hat_r = pi2hat_r[1:]
    pi3hat_r = pi3hat_r[1:]
    w2_r = w2_r[1:]
    da1_r = da1_r[1:]
    da2_r = da2_r[1:]

    mu1hat_a = mu1hat_a[1:]
    pi1hat_a = pi1hat_a[1:]
    pi2hat_a = pi2hat_a[1:]
    pi3hat_a = pi3hat_a[1:]
    w2_a = w2_a[1:]
    da1_a = da1_a[1:]
    da2_a = da2_a[1:]

    sa1_r = mu1_r * (1.0 - mu1_r)
    sa1_a = mu1_a * (1.0 - mu1_a)

    traj = {
        'mu_r': np.column_stack([mu1_r, mu2_r, mu3_r]),
        'sa_r': np.column_stack([sa1_r, 1.0 / pi2_r, 1.0 / pi3_r]),
        'mu_a': np.column_stack([mu1_a, mu2_a, mu3_a]),
        'sa_a': np.column_stack([sa1_a, 1.0 / pi2_a, 1.0 / pi3_a]),
        'muhat_r': np.column_stack([mu1hat_r, mu2hat_r, mu3hat_r]),
        'sahat_r': np.column_stack([1.0 / pi1hat_r, 1.0 / pi2hat_r, 1.0 / pi3hat_r]),
        'muhat_a': np.column_stack([mu1hat_a, mu2hat_a, mu3hat_a]),
        'sahat_a': np.column_stack([1.0 / pi1hat_a, 1.0 / pi2hat_a, 1.0 / pi3hat_a]),
        'w_r': w2_r,
        'da_r': np.column_stack([da1_r, da2_r]),
        'w_a': w2_a,
        'da_a': np.column_stack([da1_a, da2_a]),
    }

    inf_states = np.full((n - 1, 3, 4), np.nan)
    inf_states[:, :, 0] = traj['muhat_r']
    inf_states[:, :, 1] = traj['sahat_r']
    inf_states[:, :, 2] = traj['muhat_a']
    inf_states[:, :, 3] = traj['sahat_a']

    return [traj, inf_states]
