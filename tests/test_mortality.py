# -*- coding: utf-8 -*-
"""Проверки модуля смертности (не требуют torch)."""
import pytest

from deep_hedging.config import Config, zcb0
from deep_hedging.mortality import (QX_POP, annual_premium, annuity_due, kpx,
                                    _m_to_q, qx)


def test_m_to_q_matches_actuarial_relation():
    # q = m/(1+0.5m); для m=5.6 промилле ожидаем 0.005584
    assert _m_to_q(5.6) == pytest.approx(0.005584, abs=1e-6)


def test_qx_below_mx():
    # вероятность смерти всегда меньше возрастного коэффициента
    for age in range(20, 100):
        assert 0 < QX_POP[age] < 1


def test_qx_increases_with_age():
    ages = sorted(QX_POP)
    vals = [QX_POP[a] for a in ages]
    assert vals == sorted(vals)


def test_select_factor_lowers_mortality():
    base = Config()
    sel = base.with_(select_factor=0.5)
    assert qx(40, sel) < qx(40, base)
    assert kpx(base.T, sel) > kpx(base.T, base)


def test_survival_probability_reference():
    # 10p40 при популяционной смертности Росстата 2023
    assert kpx(10, Config()) == pytest.approx(0.9380, abs=5e-4)


def test_equivalence_principle():
    # P_ann * a_due должно в точности воспроизводить единовременную премию
    cfg = Config()
    assert annual_premium(cfg) * annuity_due(cfg) == pytest.approx(cfg.P0_single)


def test_zcb_reduces_to_flat_curve():
    # при малой sig_r выпуклостная поправка ничтожна
    cfg = Config(sig_r=1e-9)
    from math import exp
    assert zcb0(cfg, 1.0) == pytest.approx(exp(-cfg.r0), rel=1e-6)


def test_unknown_age_raises():
    with pytest.raises(KeyError):
        qx(5, Config())
