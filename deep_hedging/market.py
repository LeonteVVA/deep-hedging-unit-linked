# -*- coding: utf-8 -*-
"""Симуляция рыночных факторов: недвижимость, прокси, процентная ставка."""
from math import exp, sqrt

import torch

from .config import Config


def bond_price(cfg: Config, t: float, rt: torch.Tensor, Tt: float) -> torch.Tensor:
    """Аналитическая цена бескупонной облигации Hull-White в момент t."""
    B = (1 - exp(-cfg.a_hw * (Tt - t))) / cfg.a_hw
    term = ((B - (Tt - t)) * (cfg.a_hw * cfg.theta_hw - 0.5 * cfg.sig_r ** 2)
            / cfg.a_hw ** 2 - (cfg.sig_r ** 2 * B ** 2) / (4 * cfg.a_hw))
    return exp(term) * torch.exp(-B * rt)


def simulate(cfg: Config, M: int, device) -> dict:
    """Траектории всех факторов модели.

    Торгуемые инструменты (прокси, облигация) симулируются с риск-нейтральным
    сносом: дисконтированные цены торгуемых активов обязаны быть мартингалами.
    Нехеджируемый скачок незавершения несёт физическую неопределённость.

    При ``eta_S = 1`` скачок не влияет на цену базового актива: банкротство
    отдельного застройщика не обрушивает рыночный индекс жилья, а прекращает
    договор. Влияние события реализуется только через выплату.

    Возвращает словарь с ключами ``S, H, rate, Dfac, theta``.
    """
    N, dt = cfg.N, cfg.dt
    Z = torch.randn(M, N, 3, device=device)
    Zs = Z[:, :, 0]
    Zh = cfg.rho_H * Z[:, :, 0] + sqrt(max(1 - cfg.rho_H ** 2, 0.0)) * Z[:, :, 1]
    Zr = cfg.rho_Sr * Z[:, :, 0] + sqrt(max(1 - cfg.rho_Sr ** 2, 0.0)) * Z[:, :, 2]

    logS = torch.zeros(M, N + 1, device=device)
    logH = torch.zeros(M, N + 1, device=device)
    rate = torch.zeros(M, N + 1, device=device)
    rate[:, 0] = cfg.r0
    integ = torch.zeros(M, N + 1, device=device)

    jc = cfg.lam * (1 - cfg.eta_S)            # компенсатор скачка
    one = torch.tensor(1.0, device=device)
    et = torch.tensor(float(cfg.eta_S), device=device)
    theta = torch.full((M,), N + 1, dtype=torch.long, device=device)
    sig_h = cfg.sigma if abs(cfg.rho_H - 1.0) < 1e-9 else cfg.sig_H
    p_jump = 1 - exp(-cfg.lam * dt)

    for i in range(N):
        jump = torch.rand(M, device=device) < p_jump
        first = jump & (theta > N)
        theta[first] = i + 1                   # скачок на интервале (t_i, t_{i+1}]
        ri = rate[:, i]
        logS[:, i + 1] = (logS[:, i] + (ri - 0.5 * cfg.sigma ** 2 + jc) * dt
                          + cfg.sigma * sqrt(dt) * Zs[:, i]
                          + torch.log(torch.where(jump, et, one)))
        logH[:, i + 1] = (logH[:, i] + (ri - 0.5 * sig_h ** 2) * dt
                          + sig_h * sqrt(dt) * Zh[:, i])
        rate[:, i + 1] = ri + (cfg.theta_hw - cfg.a_hw * ri) * dt + cfg.sig_r * sqrt(dt) * Zr[:, i]
        integ[:, i + 1] = integ[:, i] + ri * dt

    return dict(S=cfg.S0 * torch.exp(logS), H=torch.exp(logH),
                rate=rate, Dfac=torch.exp(-integ), theta=theta)


def sample_death(cfg: Config, M: int, device) -> torch.Tensor:
    """Год смерти K (выплата в конце года K); K > T означает дожитие."""
    from .mortality import qx
    alive = torch.ones(M, dtype=torch.bool, device=device)
    K = torch.full((M,), cfg.T + 1, dtype=torch.long, device=device)
    for k in range(cfg.T):
        die = (torch.rand(M, device=device) < qx(cfg.x_age + k, cfg)) & alive
        K[die] = k + 1
        alive = alive & (~die)
    return K
