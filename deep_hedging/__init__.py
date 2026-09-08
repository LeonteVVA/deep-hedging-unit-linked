# -*- coding: utf-8 -*-
"""Расчёт справедливой стоимости участия в прибыли методом Deep Hedging.

Паевой договор страхования жизни со стратегией инвестирования в строящуюся
недвижимость; оценка на неполном рынке через энтропийную меру риска.
"""
from .config import Config, zcb0
from .data import Calibration, calibrate, geltner_unsmooth, load_prices
from .mortality import annual_premium, annuity_due, describe, kpx, qx
from .verification import bs_call, deep_hedging_complete_market, mc_call

from .contract import (ANNUAL, SINGLE, contributions, fund_and_guarantee,
                       payoff, premium_flow)
from .market import bond_price, sample_death, simulate
from .pricing import (HedgeNet, entropic, evaluate, get_device, price, train,
                      v_part)

from . import config, contract, data, market, mortality, pricing, verification

__all__ = [
    # параметры
    "Config", "zcb0",
    # данные и калибровка
    "Calibration", "calibrate", "load_prices", "geltner_unsmooth",
    # смертность
    "qx", "kpx", "annuity_due", "annual_premium", "describe",
    # рынок
    "simulate", "sample_death", "bond_price",
    # договор
    "SINGLE", "ANNUAL", "fund_and_guarantee", "contributions",
    "premium_flow", "payoff",
    # оценка
    "HedgeNet", "entropic", "train", "evaluate", "price", "v_part",
    "get_device",
    # верификация
    "bs_call", "mc_call", "deep_hedging_complete_market",
    # подмодули
    "config", "contract", "data", "market", "mortality", "pricing",
    "verification",
]
