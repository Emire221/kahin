# The Oracle - Models Package
"""
Models modülü: ML modelleri (Dixon-Coles, XGBoost) ve eğitim scriptleri

Bu modül aşağıdaki ana bileşenleri içerir:
- DixonColesModel: Poisson tabanlı skor tahmin modeli
- XGBoostPredictor: Makine öğrenmesi tabanlı sonuç tahmincisi
- Trainer: Model eğitim ve kaydetme işlemleri
"""

from .dixon_coles import DixonColesModel
from .xgboost_model import XGBoostPredictor

__all__ = ["DixonColesModel", "XGBoostPredictor"]
