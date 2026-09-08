# -*- coding: utf-8 -*-
"""Загрузка индекса цен жилья и устранение сглаживания по Гелтнеру."""
import os
from math import sqrt
from dataclasses import dataclass

import numpy as np

DEFAULT_FILE = "realty_index.csv"
FREQ = 4  # квартальные данные


@dataclass
class Calibration:
    """Результат калибровки волатильности по историческому ряду."""
    prices: np.ndarray
    is_real: bool
    phi: float           # коэффициент сглаживания
    sigma_obs: float     # наблюдаемая волатильность, годовая
    sigma_true: float    # десглаженная волатильность, годовая
    mu: float            # оценка сноса, годовая

    @property
    def uplift(self) -> float:
        return self.sigma_true / self.sigma_obs

    def summary(self) -> str:
        src = "реальные данные" if self.is_real else "СИНТЕТИКА"
        return (f"{src}, {len(self.prices)} наблюдений; phi={self.phi:.3f}; "
                f"sigma {self.sigma_obs:.4f} -> {self.sigma_true:.4f} "
                f"(x{self.uplift:.2f}); mu={self.mu:.4f}")


def load_prices(path: str = DEFAULT_FILE):
    """Читает ряд уровней цен. При отсутствии файла возвращает синтетику."""
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        col = "price" if "price" in df.columns else df.columns[-1]
        return df[col].to_numpy(float), True
    rng = np.random.default_rng(42)
    n = 60
    tr = rng.normal(0.02, 0.06, n)
    obs = np.zeros(n)
    obs[0] = tr[0]
    for t in range(1, n):
        obs[t] = 0.6 * obs[t - 1] + 0.4 * tr[t]
    return 100 * np.exp(np.cumsum(obs)), False


def geltner_unsmooth(prices, phi=None):
    """Десглаживание Гелтнера (1993).

    Оценочные индексы недвижимости занижают волатильность из-за усреднения и
    запаздывания оценок. Наблюдаемая доходность моделируется как
    ``r_obs[t] = phi*r_obs[t-1] + (1-phi)*r_true[t]``; обращая соотношение,
    восстанавливаем истинный ряд.

    Возвращает ``(sigma_obs_q, sigma_true_q, phi, mu_q)`` в квартальных единицах.
    """
    lr = np.diff(np.log(prices))
    if phi is None:
        phi = np.cov(lr[:-1], lr[1:], bias=True)[0, 1] / np.var(lr[:-1])
        phi = float(np.clip(phi, -0.99, 0.99))
    r_true = (lr[1:] - phi * lr[:-1]) / (1 - phi)
    return lr.std(ddof=1), r_true.std(ddof=1), float(phi), lr.mean()


def calibrate(path: str = DEFAULT_FILE, phi=None, freq: int = FREQ) -> Calibration:
    """Полная калибровка волатильности по ряду цен."""
    prices, is_real = load_prices(path)
    so, st, phi_hat, mu_q = geltner_unsmooth(prices, phi)
    return Calibration(prices=prices, is_real=is_real, phi=phi_hat,
                       sigma_obs=so * sqrt(freq), sigma_true=st * sqrt(freq),
                       mu=mu_q * freq)
