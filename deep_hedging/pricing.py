# -*- coding: utf-8 -*-
"""Глубокое хеджирование: сеть, обучение, out-of-sample оценка."""
from math import log

import torch
import torch.nn as nn

from .config import Config
from .contract import fund_and_guarantee, payoff, SINGLE
from .market import bond_price, sample_death, simulate


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class HedgeNet(nn.Module):
    """Стратегия хеджирования: (t, logH, logS, logF, r) -> позиции в 2 активах.

    Веса разделены по времени, момент подаётся признаком: это сокращает число
    параметров и улучшает обобщение.
    """

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, t, logH, logS, logF, rate):
        return self.net(torch.stack([t, logH, logS, logF, rate], -1))


def entropic(X: torch.Tensor, gamma: float) -> torch.Tensor:
    """Энтропийная мера риска (1/gamma) log E[exp(-gamma X)]."""
    return (1 / gamma) * torch.logsumexp(-gamma * X, 0) - (1 / gamma) * log(X.shape[0])


def rollout(net, cfg: Config, mode: str, alpha: float, M: int, device,
            return_at_maturity: bool = False) -> torch.Tensor:
    """Один прогон: возвращает позицию P&L = торговый результат минус выплата."""
    N, dt = cfg.N, cfg.dt
    times = torch.arange(N, device=device).float() / N
    mkt = simulate(cfg, M, device)
    S, H, rate, Dfac, theta = mkt["S"], mkt["H"], mkt["rate"], mkt["Dfac"], mkt["theta"]
    K = sample_death(cfg, M, device)

    F, _ = fund_and_guarantee(cfg, S, Dfac, mode)
    Htil = H * Dfac
    bond = torch.stack([bond_price(cfg, i * dt, rate[:, i], cfg.T)
                        for i in range(N + 1)], dim=1)
    Btil = bond * Dfac

    gains = torch.zeros(M, device=device)
    for i in range(N):
        d = net(times[i].expand(M), torch.log(H[:, i]), torch.log(S[:, i]),
                torch.log(F[:, i].clamp(min=1e-6)), rate[:, i])
        gains = (gains + d[:, 0] * (Htil[:, i + 1] - Htil[:, i])
                 + d[:, 1] * (Btil[:, i + 1] - Btil[:, i]))

    xi = payoff(cfg, S, Dfac, theta, K, mode, alpha, return_at_maturity)
    return gains - xi


def train(cfg: Config, mode: str = SINGLE, alpha: float = None, device=None,
          return_at_maturity: bool = False, verbose: bool = False):
    """Обучение стратегии минимизацией энтропийной меры риска."""
    device = device or get_device()
    alpha = cfg.alpha if alpha is None else alpha
    torch.manual_seed(cfg.seed)
    net = HedgeNet(cfg.hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    net.train()
    for ep in range(cfg.epochs):
        X = rollout(net, cfg, mode, alpha, cfg.M, device, return_at_maturity)
        loss = entropic(X, cfg.gamma_risk)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 50 == 0:
            print(f"  эпоха {ep + 1}/{cfg.epochs}: loss={loss.item():.5f}")
    return net


def evaluate(net, cfg: Config, mode: str = SINGLE, alpha: float = None, device=None,
             return_at_maturity: bool = False, return_pnl: bool = False,
             chunk: int = 20000):
    """Цена вне обучающей выборки: отдельный прогон без градиента."""
    device = device or get_device()
    alpha = cfg.alpha if alpha is None else alpha
    net.eval()
    torch.manual_seed(cfg.eval_seed)
    parts, done = [], 0
    with torch.no_grad():
        while done < cfg.M_eval:
            m = min(chunk, cfg.M_eval - done)
            parts.append(rollout(net, cfg, mode, alpha, m, device, return_at_maturity))
            done += m
    X = torch.cat(parts)
    price = entropic(X, cfg.gamma_risk).item()
    return (price, X.cpu().numpy()) if return_pnl else price


def price(cfg: Config, mode: str = SINGLE, alpha: float = None, device=None,
          return_at_maturity: bool = False) -> float:
    """Цена обязательства: обучение + оценка."""
    net = train(cfg, mode, alpha, device, return_at_maturity)
    return evaluate(net, cfg, mode, alpha, device, return_at_maturity)


def v_part(cfg: Config, mode: str = SINGLE, device=None,
           return_at_maturity: bool = False, return_parts: bool = False):
    """Справедливая стоимость участия V_part = pi(xi_alpha) - pi(xi_0).

    Обе цены считаются на общих случайных числах (одинаковый ``seed``), что
    существенно снижает дисперсию разности.
    """
    pa = price(cfg, mode, cfg.alpha, device, return_at_maturity)
    p0 = price(cfg, mode, 0.0, device, return_at_maturity)
    if return_parts:
        return dict(pi_alpha=pa, pi_0=p0, V_part=pa - p0)
    return pa - p0
