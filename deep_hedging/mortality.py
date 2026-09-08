# -*- coding: utf-8 -*-
"""Смертность: возрастные коэффициенты Росстата -> вероятности дожития."""
from .config import Config, zcb0

MORT_YEAR = 2023

#: Возрастные коэффициенты смертности, промилле.
#: Росстат (ЕМИСС), РФ, оба пола, всё население, 2023 г.
MX_ROSSTAT = {
    (20, 24): 1.3, (25, 29): 1.9, (30, 34): 2.7, (35, 39): 3.9,
    (40, 44): 5.6, (45, 49): 7.2, (50, 54): 8.7, (55, 59): 11.6,
    (60, 64): 16.5, (65, 69): 22.6, (70, 74): 32.5, (75, 79): 44.6,
    (80, 84): 79.8, (85, 99): 156.9,
}


def _m_to_q(m_permille: float) -> float:
    """Возрастной коэффициент -> годовая вероятность смерти.

    Стандартное актуарное соотношение ``q = m / (1 + 0.5 m)``.
    """
    m = m_permille / 1000.0
    return m / (1 + 0.5 * m)


QX_POP = {age: _m_to_q(mv)
          for (lo, hi), mv in MX_ROSSTAT.items()
          for age in range(lo, hi + 1)}


def qx(age: int, cfg: Config) -> float:
    """Годовая вероятность смерти с учётом андеррайтингового отбора.

    Общенациональная смертность выше смертности застрахованных: страховщик
    отсеивает лиц с повышенным риском. Коэффициент ``select_factor`` задаёт
    отношение; значение 1.0 (популяционная смертность) консервативно.
    """
    if age not in QX_POP:
        raise KeyError(f"нет q_x для возраста {age}; дополните MX_ROSSTAT")
    return min(cfg.select_factor * QX_POP[age], 1.0)


def kpx(k: int, cfg: Config) -> float:
    """Вероятность дожития застрахованного возраста ``x_age`` ещё k лет."""
    p = 1.0
    for j in range(k):
        p *= (1 - qx(cfg.x_age + j, cfg))
    return p


def annuity_due(cfg: Config) -> float:
    """Аннуитет дожития: сумма P(0,k) * kpx по ценам облигаций Hull-White."""
    return sum(zcb0(cfg, k) * kpx(k, cfg) for k in range(cfg.T))


def annual_premium(cfg: Config) -> float:
    """Эквивалентный ежегодный взнос из принципа эквивалентности премий."""
    return cfg.P0_single / annuity_due(cfg)


def describe(cfg: Config) -> str:
    return (f"Росстат {MORT_YEAR}, оба пола, отбор {cfg.select_factor}; "
            f"{cfg.T}p{cfg.x_age}={kpx(cfg.T, cfg):.4f}; "
            f"a_due={annuity_due(cfg):.4f}; P_ann={annual_premium(cfg):.4f}")
