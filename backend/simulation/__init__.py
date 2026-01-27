# The Oracle - Simulation Package
"""
Simulation modülü: Backtest, Sanal Kasa ve Value Bet hesaplamaları

Bu modül, modellerin geçmiş performansını test etmek ve
sanal bahis simülasyonu yapmak için kullanılır.
"""

from .wallet import Wallet
from .backtest_engine import BacktestEngine

__all__ = ["Wallet", "BacktestEngine"]
