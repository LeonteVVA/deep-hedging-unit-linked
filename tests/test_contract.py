# -*- coding: utf-8 -*-
"""Проверки структуры договора. Требуют torch."""
import pytest

torch = pytest.importorskip("torch")

from deep_hedging.config import Config
from deep_hedging.config import Config  # noqa: F401
from deep_hedging.contract import (ANNUAL, SINGLE, contributions,
                                   fund_and_guarantee, payoff, premium_flow)


def _flat_market(cfg, M=8):
    """Детерминированный рынок: S = 1, ставка = r0, без скачков."""
    N = cfg.N
    S = torch.ones(M, N + 1)
    t = torch.arange(N + 1).float() * cfg.dt
    Dfac = torch.exp(-cfg.r0 * t).expand(M, N + 1).contiguous()
    theta = torch.full((M,), N + 1, dtype=torch.long)
    K = torch.full((M,), cfg.T + 1, dtype=torch.long)
    return S, Dfac, theta, K


def test_single_fund_starts_at_invested_premium():
    cfg = Config()
    S, Dfac, _, _ = _flat_market(cfg)
    F, G = fund_and_guarantee(cfg, S, Dfac, SINGLE)
    assert F[0, 0].item() == pytest.approx(cfg.P0_single)
    assert G[0, 0].item() == pytest.approx(cfg.P0_single)


def test_gamma_inv_zero_gives_riskfree_growth():
    # при нулевой доле в недвижимости фонд растёт строго по безрисковой ставке
    cfg = Config(gamma_inv=0.0)
    S, Dfac, _, _ = _flat_market(cfg)
    F, _ = fund_and_guarantee(cfg, S, Dfac, SINGLE)
    from math import exp
    assert F[0, cfg.N].item() == pytest.approx(cfg.P0_single * exp(cfg.r0 * cfg.T), rel=1e-4)


def test_premium_flow_zero_for_single():
    cfg = Config()
    S, Dfac, theta, K = _flat_market(cfg)
    assert torch.all(premium_flow(cfg, Dfac, theta, K, SINGLE) == 0)


def test_premium_flow_positive_for_annual():
    cfg = Config()
    S, Dfac, theta, K = _flat_market(cfg)
    flow = premium_flow(cfg, Dfac, theta, K, ANNUAL)
    assert torch.all(flow > 0)


def test_premium_flow_stops_after_failure():
    # если стройка встала на первом шаге, будущие взносы не уплачиваются
    cfg = Config()
    S, Dfac, theta, K = _flat_market(cfg)
    theta = torch.ones_like(theta)
    assert torch.all(premium_flow(cfg, Dfac, theta, K, ANNUAL) == 0)


def test_participation_increases_payoff():
    cfg = Config(gamma_inv=0.0)          # фонд обгоняет гарантию: r0 > g
    S, Dfac, theta, K = _flat_market(cfg)
    lo = payoff(cfg, S, Dfac, theta, K, SINGLE, 0.0)
    hi = payoff(cfg, S, Dfac, theta, K, SINGLE, 1.0)
    assert torch.all(hi >= lo)
    assert (hi - lo).abs().max().item() > 0


def test_escrow_return_discounted_at_failure_moment():
    # возврат в момент theta дороже возврата в момент выплаты
    cfg = Config()
    S, Dfac, theta, K = _flat_market(cfg)
    theta = torch.full_like(theta, 12)   # незавершение через год
    early = payoff(cfg, S, Dfac, theta, K, SINGLE, cfg.alpha)
    late = payoff(cfg, S, Dfac, theta, K, SINGLE, cfg.alpha, return_at_maturity=True)
    assert torch.all(early > late)


def test_contributions_capped_at_term():
    cfg = Config()
    months = torch.tensor([0, 12, 60, 119, cfg.N])
    c = contributions(cfg, ANNUAL, months)
    assert c[-1].item() == pytest.approx(c[-2].item())   # не более T взносов
