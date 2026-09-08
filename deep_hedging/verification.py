# -*- coding: utf-8 -*-
"""Независимые проверки: Блэк-Шоулз и Монте-Карло."""
from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm


def bs_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Аналитическая цена европейского колл-опциона."""
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)


def mc_call(S0: float, K: float, r: float, sigma: float, T: float,
            lam: float = 0.0, eta: float = 1.0, M: int = 200_000,
            n: int = 30, seed: int = 0):
    """Монте-Карло для колл-опциона на актив со скачком.

    Возвращает ``(цена, стандартная ошибка)``.
    """
    rng = np.random.default_rng(seed)
    dt = T / n
    lg = np.full(M, log(S0))
    jc = lam * (1 - eta)
    for _ in range(n):
        Z = rng.standard_normal(M)
        jp = rng.random(M) < (1 - exp(-lam * dt))
        lg += (r - 0.5 * sigma ** 2 + jc) * dt + sigma * sqrt(dt) * Z
        lg[jp] += log(eta)
    pay = np.maximum(np.exp(lg) - K, 0.0)
    return exp(-r * T) * pay.mean(), exp(-r * T) * pay.std(ddof=1) / sqrt(M)


def deep_hedging_complete_market(cfg, Tv: float = 1.0, gamma: float = 0.5,
                                 device=None):
    """Цена колл-опциона методом глубокого хеджирования на ПОЛНОМ рынке.

    Упрощённая конфигурация (один фактор, чистое ГБД, rho_H = 1, lambda = 0)
    изолирует именно механизм глубокого хеджирования. Совпадение с формулой
    Блэка-Шоулза подтверждает корректность метода оценки.
    """
    import torch
    import torch.nn as nn
    from .pricing import entropic, get_device

    device = device or get_device()
    n = int(Tv * cfg.steps_per_year)
    dt = cfg.dt
    sigma, r = cfg.sigma, cfg.r0

    net = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    times = torch.arange(n, device=device).float() / n

    def roll(M):
        Z = torch.randn(M, n, device=device)
        logS = torch.zeros(M, n + 1, device=device)
        for i in range(n):
            logS[:, i + 1] = (logS[:, i] + (r - 0.5 * sigma ** 2) * dt
                              + sigma * sqrt(dt) * Z[:, i])
        S = torch.exp(logS)
        Stil = S * torch.exp(-r * torch.arange(n + 1, device=device).float() * dt)
        gains = torch.zeros(M, device=device)
        for i in range(n):
            d = net(torch.stack([times[i].expand(M), torch.log(S[:, i])], -1))[:, 0]
            gains = gains + d * (Stil[:, i + 1] - Stil[:, i])
        xi = torch.clamp(S[:, n] - 1.0, min=0.0) * exp(-r * Tv)
        return gains - xi

    torch.manual_seed(0)
    for _ in range(cfg.epochs):
        loss = entropic(roll(cfg.M), gamma)
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.manual_seed(99999)
    parts, done = [], 0
    with torch.no_grad():
        while done < cfg.M_eval:
            m = min(20000, cfg.M_eval - done)
            parts.append(roll(m))
            done += m
    return entropic(torch.cat(parts), gamma).item()
