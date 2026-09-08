# -*- coding: utf-8 -*-
"""Согласованность Монте-Карло с формулой Блэка-Шоулза (не требует torch)."""
import pytest

from deep_hedging.verification import bs_call, mc_call


def test_mc_matches_black_scholes():
    p, se = mc_call(1.0, 1.0, 0.05, 0.0879, 1.0, lam=0.0, M=200_000, seed=0)
    bs = bs_call(1.0, 1.0, 0.05, 0.0879, 1.0)
    assert abs(p - bs) < 4 * se        # в пределах учетверённой стандартной ошибки


def test_no_jump_when_eta_is_one():
    # при eta=1 скачок не влияет на динамику: цены со скачком и без совпадают
    a, _ = mc_call(1.0, 1.0, 0.05, 0.0879, 1.0, lam=0.00, eta=1.0, seed=7)
    b, _ = mc_call(1.0, 1.0, 0.05, 0.0879, 1.0, lam=0.02, eta=1.0, seed=7)
    assert a == pytest.approx(b, abs=1e-12)


def test_call_price_increases_with_volatility():
    lo = bs_call(1.0, 1.0, 0.05, 0.05, 1.0)
    hi = bs_call(1.0, 1.0, 0.05, 0.15, 1.0)
    assert hi > lo
