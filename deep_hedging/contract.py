# -*- coding: utf-8 -*-
"""Структура договора: фонд, гарантия, взносы, выплата."""
import torch

from .config import Config
from .mortality import annual_premium

SINGLE = "single"
ANNUAL = "annual"


def fund_and_guarantee(cfg: Config, S, Dfac, mode: str):
    """Инвестиционный фонд F_t и гарантированный уровень G_t.

    Инвестиционная стратегия: доля ``gamma_inv`` транша размещается в
    недвижимость, остаток — в безрисковый актив (счёт B_t = 1/D(t)). Это
    непрерывный аналог коэффициента gamma базовой биномиальной модели,
    где u_gamma = gamma*u + (1-gamma)*(1+r).
    """
    gi = cfg.gamma_inv
    M, N, dt = S.shape[0], cfg.N, cfg.dt
    device = S.device
    tg = torch.arange(N + 1, device=device).float() * dt

    if mode == SINGLE:
        risky = S / S[:, 0:1]
        safe = Dfac[:, 0:1] / Dfac                       # B_t / B_0
        F = cfg.P0_single * (gi * risky + (1 - gi) * safe)
        G = (cfg.P0_single * torch.exp(cfg.g * tg)).expand(M, N + 1)
        return F, G

    P_ann = annual_premium(cfg)
    F = torch.zeros(M, N + 1, device=device)
    G = torch.zeros(M, N + 1, device=device)
    for k in range(cfg.T):
        cm = k * cfg.steps_per_year
        ratio = torch.zeros(M, N + 1, device=device)
        # транш куплен в момент cm и входит в фонд со следующего шага:
        # взнос, уплачиваемый ровно в момент выплаты, не начисляется
        risky = S[:, cm + 1:] / S[:, cm:cm + 1]
        safe = Dfac[:, cm:cm + 1] / Dfac[:, cm + 1:]
        ratio[:, cm + 1:] = P_ann * (gi * risky + (1 - gi) * safe)
        F = F + ratio
        tcm = cm * dt
        G = G + torch.where(tg > tcm, P_ann * torch.exp(cfg.g * (tg - tcm)),
                            torch.zeros_like(tg)).expand(M, N + 1)
    return F, G


def contributions(cfg: Config, mode: str, months: torch.Tensor) -> torch.Tensor:
    """Сумма фактически внесённых взносов к указанному моменту."""
    if mode == SINGLE:
        return torch.full_like(months, cfg.P0_single, dtype=torch.float)
    n = torch.clamp(months // cfg.steps_per_year + 1, max=cfg.T)
    return annual_premium(cfg) * n.float()


def premium_flow(cfg: Config, Dfac, theta, K, mode: str) -> torch.Tensor:
    """Дисконтированный поток будущих взносов (k = 1..T-1).

    Уплата условна: взносы прекращаются при смерти застрахованного и при
    незавершении строительства, поскольку договор в инвестиционной части
    прекращается. Для единовременной премии поток тождественно нулевой.
    """
    M = Dfac.shape[0]
    if mode == SINGLE:
        return torch.zeros(M, device=Dfac.device)
    P_ann = annual_premium(cfg)
    total = torch.zeros(M, device=Dfac.device)
    for k in range(1, cfg.T):
        cm = k * cfg.steps_per_year
        paid = (K > k) & (theta > cm)
        total = total + P_ann * Dfac[:, cm] * paid.float()
    return total


def payoff(cfg: Config, S, Dfac, theta, K, mode: str, alpha: float,
           return_at_maturity: bool = False) -> torch.Tensor:
    """Дисконтированная НЕТТО-выплата: выплата минус будущие взносы.

    Схема участия — терминальный бонус (point-to-point): превышение фонда над
    гарантией оценивается однократно, в момент выплаты.

    При незавершении строительства средства возвращаются В МОМЕНТ theta, а не
    в момент выплаты по договору (счёт эскроу беспроцентный). Флаг
    ``return_at_maturity`` включает контрфактный вариант для декомпозиции.
    """
    M = S.shape[0]
    device = S.device
    idx = torch.arange(M, device=device)
    F, G = fund_and_guarantee(cfg, S, Dfac, mode)

    pm = torch.where(K <= cfg.T, K * cfg.steps_per_year,
                     torch.full_like(K, cfg.N)).clamp(max=cfg.N).long()
    Phi = G[idx, pm] + alpha * torch.clamp(F[idx, pm] - G[idx, pm], min=0.0)
    if cfg.SA > 0:
        Phi = torch.where(K <= cfg.T, torch.clamp(Phi, min=cfg.SA), Phi)

    th = theta.clamp(max=cfg.N)
    ret = cfg.eta_R * contributions(cfg, mode, th)
    failed = theta <= pm
    if return_at_maturity:
        xi = torch.where(failed, ret, Phi) * Dfac[idx, pm]
    else:
        xi = torch.where(failed, ret * Dfac[idx, th], Phi * Dfac[idx, pm])

    return xi - premium_flow(cfg, Dfac, theta, K, mode)
