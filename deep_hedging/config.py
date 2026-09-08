# -*- coding: utf-8 -*-
"""Параметры модели. Один объект вместо глобальных переменных."""
from dataclasses import dataclass, replace
from math import exp


@dataclass(frozen=True)
class Config:
    """Полный набор параметров расчёта.

    Изменение параметра — через ``cfg.with_(sigma=0.2)``: возвращается новая
    конфигурация, исходная не мутируется. Это заменяет подмену глобальных
    переменных в стресс-тестах.
    """

    # --- рынок ---
    sigma: float = 0.0879          # десглаженная волатильность недвижимости
    S0: float = 1.0
    T: int = 10                    # срок договора, лет
    steps_per_year: int = 12

    # --- процентная ставка (Hull-White) ---
    r0: float = 0.05
    a_hw: float = 0.5
    sig_r: float = 0.01
    rho_Sr: float = -0.3

    # --- риск незавершения строительства (214-ФЗ, эскроу) ---
    lam: float = 0.02              # интенсивность незавершения
    eta_R: float = 1.00            # доля ВОЗВРАТА средств (эскроу: 1.0)
    eta_S: float = 1.00            # доля стоимости АКТИВА при скачке

    # --- хеджирующий прокси ---
    rho_H: float = 0.8             # корреляция прокси с недвижимостью
    sig_H_add: float = 0.02        # надбавка к волатильности прокси

    # --- инвестиционная стратегия ---
    gamma_inv: float = 1.00        # доля фонда в недвижимости

    # --- контракт ---
    x_age: int = 40
    g: float = 0.02                # гарантированная техническая ставка
    alpha: float = 0.6             # доля участия в прибыли
    beta: float = 0.95             # доля премии в инвестиционном фонде
    P: float = 1.0                 # единовременная премия (нормировка)
    SA: float = 0.0                # минимальная страховая сумма по смерти

    # --- смертность ---
    select_factor: float = 1.00    # андеррайтинговый отбор (1.0 = популяционная)

    # --- мера риска и обучение ---
    gamma_risk: float = 2.0        # неприятие риска (энтропийная мера)
    M: int = 15000                 # траекторий в батче
    epochs: int = 200
    lr: float = 1e-3
    hidden: int = 64
    M_eval: int = 200_000          # траекторий для out-of-sample оценки
    seed: int = 1
    eval_seed: int = 12345

    # ---------------------------------------------------------------- производные
    @property
    def N(self) -> int:
        """Число шагов дискретизации."""
        return self.T * self.steps_per_year

    @property
    def dt(self) -> float:
        return 1.0 / self.steps_per_year

    @property
    def theta_hw(self) -> float:
        """Постоянный theta, дающий плоскую начальную кривую."""
        return self.a_hw * self.r0

    @property
    def sig_H(self) -> float:
        return self.sigma + self.sig_H_add

    @property
    def P0_single(self) -> float:
        """Инвестируемая часть единовременной премии."""
        return self.beta * self.P

    def with_(self, **kw) -> "Config":
        """Копия конфигурации с изменёнными полями."""
        return replace(self, **kw)


def zcb0(cfg: Config, t: float) -> float:
    """Цена бескупонной облигации в момент 0 (Hull-White, постоянный theta)."""
    if t <= 0:
        return 1.0
    B = (1 - exp(-cfg.a_hw * t)) / cfg.a_hw
    term = ((B - t) * (cfg.a_hw * cfg.theta_hw - 0.5 * cfg.sig_r ** 2) / cfg.a_hw ** 2
            - (cfg.sig_r ** 2 * B ** 2) / (4 * cfg.a_hw))
    return exp(term - B * cfg.r0)
