# -*- coding: utf-8 -*-
"""Проверки калибровки волатильности (не требуют torch)."""
import numpy as np
import pytest

from deep_hedging.data import geltner_unsmooth


def test_no_autocorrelation_leaves_sigma_unchanged():
    # если сглаживания нет (phi=0), десглаженная волатильность равна наблюдаемой
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.05, 400)))
    so, st, phi, _ = geltner_unsmooth(prices, phi=0.0)
    assert st == pytest.approx(so, rel=0.05)


def test_smoothing_is_detected_and_reversed():
    # строим ряд с известным сглаживанием и проверяем, что phi оценивается
    rng = np.random.default_rng(1)
    n, phi_true = 2000, 0.6
    true_r = rng.normal(0, 0.05, n)
    obs = np.zeros(n)
    for t in range(1, n):
        obs[t] = phi_true * obs[t - 1] + (1 - phi_true) * true_r[t]
    prices = 100 * np.exp(np.cumsum(obs))
    so, st, phi_hat, _ = geltner_unsmooth(prices)
    assert phi_hat == pytest.approx(phi_true, abs=0.1)
    assert st > so                     # десглаживание повышает волатильность


def test_unsmoothing_increases_volatility_on_real_pattern():
    rng = np.random.default_rng(2)
    n = 500
    true_r = rng.normal(0.005, 0.04, n)
    obs = np.zeros(n)
    for t in range(1, n):
        obs[t] = 0.7 * obs[t - 1] + 0.3 * true_r[t]
    prices = 100 * np.exp(np.cumsum(obs))
    so, st, _, _ = geltner_unsmooth(prices)
    assert st / so > 1.5
