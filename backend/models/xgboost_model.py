"""
The Oracle - XGBoost Maç Sonucu Tahmincisi

Bu modül, XGBoost (eXtreme Gradient Boosting) algoritmasını kullanarak
futbol maçlarının sonuçlarını (1-X-2) tahmin eder.

Temel Özellikler:
    - Öznitelik mühendisliği (Feature Engineering)
    - Son 5 maç form analizi
    - Ev/Deplasman performans ayrımı
    - Takım güç sıralaması (Elo benzeri)
    - Head-to-Head (H2H) istatistikleri
    - Model kalibrasyon desteği

Neden XGBoost?
    - Tablo verilerinde (tabular data) en iyi performans
    - Eksik veri toleransı
    - Aşırı öğrenme (overfitting) direnci
    - Hızlı eğitim ve tahmin

Kullanım:
    from backend.models.xgboost_model import XGBoostPredictor
    
    model = XGBoostPredictor()
    model.fit(matches_df)
    
    # Olasılık tahmini
    probs = model.predict_proba("Arsenal", "Chelsea")
    # {'home_win': 0.52, 'draw': 0.26, 'away_win': 0.22}
    
    # Öznitelik önemi
    importance = model.get_feature_importance()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import pickle
from pathlib import Path
import warnings

from loguru import logger

try:
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import cross_val_score
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost veya scikit-learn yüklü değil!")


@dataclass
class TeamStats:
    """Takım istatistikleri"""
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    form: List[str] = field(default_factory=list)  # Son maç sonuçları: W/D/L
    elo_rating: float = 1500.0
    
    @property
    def win_rate(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played
    
    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against
    
    @property
    def goals_per_game(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.goals_for / self.matches_played
    
    @property
    def goals_against_per_game(self) -> float:
        if self.matches_played == 0:
            return 0.0
        return self.goals_against / self.matches_played
    
    @property
    def form_points(self) -> float:
        """Son 5 maç form puanı (0-15 arası)"""
        if not self.form:
            return 7.5  # Ortalama
        
        points = 0
        recent = self.form[-5:]  # Son 5 maç
        for result in recent:
            if result == 'W':
                points += 3
            elif result == 'D':
                points += 1
        
        return points


class EloCalculator:
    """
    Elo Rating sistemi.
    
    Satrançtan uyarlanmış güç sıralaması sistemi.
    Güçlü takımı yenen daha çok puan kazanır.
    """
    
    def __init__(self, k_factor: float = 32.0, home_advantage: float = 100.0):
        """
        Args:
            k_factor: Güncelleme katsayısı (volatilite)
            home_advantage: İç saha Elo avantajı
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = defaultdict(lambda: 1500.0)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Beklenen skor (0-1 arası)"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update(
        self, 
        home_team: str, 
        away_team: str, 
        home_goals: int, 
        away_goals: int
    ) -> Tuple[float, float]:
        """
        Maç sonucuna göre Elo ratinglerini günceller.
        
        Returns:
            Tuple: (yeni home rating, yeni away rating)
        """
        # Mevcut ratingler
        home_rating = self.ratings[home_team] + self.home_advantage
        away_rating = self.ratings[away_team]
        
        # Beklenen skorlar
        home_expected = self.expected_score(home_rating, away_rating)
        away_expected = 1 - home_expected
        
        # Gerçek skor (1: kazandı, 0.5: beraberlik, 0: kaybetti)
        if home_goals > away_goals:
            home_actual, away_actual = 1.0, 0.0
        elif home_goals < away_goals:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        
        # Rating güncelleme
        goal_diff = abs(home_goals - away_goals)
        multiplier = np.log(max(goal_diff, 1) + 1)  # Gol farkına göre ağırlık
        
        self.ratings[home_team] += self.k_factor * multiplier * (home_actual - home_expected)
        self.ratings[away_team] += self.k_factor * multiplier * (away_actual - away_expected)
        
        return self.ratings[home_team], self.ratings[away_team]
    
    def get_rating(self, team: str) -> float:
        return self.ratings[team]


class XGBoostPredictor:
    """
    XGBoost Maç Sonucu Tahmincisi
    
    Bu model, gradient boosting kullanarak maç sonuçlarını
    (1: Ev kazanır, X: Beraberlik, 2: Deplasman kazanır) tahmin eder.
    
    Attributes:
        model: XGBoost sınıflandırıcı
        is_fitted (bool): Model eğitilmiş mi?
        feature_names (List[str]): Öznitelik isimleri
        
    Example:
        >>> model = XGBoostPredictor()
        >>> model.fit(df)
        >>> probs = model.predict_proba("Arsenal", "Chelsea")
    """
    
    # Öznitelik listesi
    FEATURE_NAMES = [
        'home_elo',
        'away_elo',
        'elo_diff',
        'home_form',
        'away_form',
        'form_diff',
        'home_goals_scored_avg',
        'away_goals_scored_avg',
        'home_goals_conceded_avg',
        'away_goals_conceded_avg',
        'home_win_rate',
        'away_win_rate',
        'h2h_home_wins',
        'h2h_away_wins',
        'h2h_draws',
        'home_home_form',  # Ev sahibinin evdeki performansı
        'away_away_form',  # Deplasmanın dışarıdaki performansı
        'home_days_rest',
        'away_days_rest',
    ]
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_calibration: bool = True,
        random_state: int = 42
    ) -> None:
        """
        XGBoost modelini başlatır.
        
        Args:
            n_estimators: Ağaç sayısı
            max_depth: Maksimum ağaç derinliği
            learning_rate: Öğrenme oranı
            min_child_weight: Minimum yaprak ağırlığı
            subsample: Satır örnekleme oranı
            colsample_bytree: Sütun örnekleme oranı
            use_calibration: Olasılık kalibrasyonu kullan
            random_state: Rastgelelik seed'i
        """
        if not HAS_XGBOOST:
            raise ImportError(
                "XGBoost ve scikit-learn gerekli! "
                "pip install xgboost scikit-learn"
            )
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.use_calibration = use_calibration
        self.random_state = random_state
        
        self._base_model = None
        self.model = None
        self.is_fitted = False
        self.feature_names = self.FEATURE_NAMES.copy()
        
        # İstatistik hesaplayıcılar
        self._team_stats: Dict[str, TeamStats] = defaultdict(TeamStats)
        self._home_stats: Dict[str, TeamStats] = defaultdict(TeamStats)  # Ev istatistikleri
        self._away_stats: Dict[str, TeamStats] = defaultdict(TeamStats)  # Deplasman istatistikleri
        self._h2h_stats: Dict[Tuple[str, str], Dict] = {}
        self._elo = EloCalculator()
        self._last_match_dates: Dict[str, datetime] = {}
        self._label_encoder = LabelEncoder()
        
        # Eğitim verisi (tahminde kullanmak için)
        self._training_df: Optional[pd.DataFrame] = None
        
        logger.info("XGBoost modeli başlatıldı")
    
    def _update_team_stats(
        self,
        team: str,
        goals_for: int,
        goals_against: int,
        is_home: bool,
        match_date: datetime
    ) -> None:
        """Takım istatistiklerini günceller"""
        stats = self._team_stats[team]
        
        stats.matches_played += 1
        stats.goals_for += goals_for
        stats.goals_against += goals_against
        
        if goals_for > goals_against:
            stats.wins += 1
            stats.points += 3
            stats.form.append('W')
        elif goals_for < goals_against:
            stats.losses += 1
            stats.form.append('L')
        else:
            stats.draws += 1
            stats.points += 1
            stats.form.append('D')
        
        # Ev/Deplasman istatistiklerini güncelle
        if is_home:
            home_stats = self._home_stats[team]
            home_stats.matches_played += 1
            home_stats.goals_for += goals_for
            home_stats.goals_against += goals_against
            if goals_for > goals_against:
                home_stats.wins += 1
            elif goals_for < goals_against:
                home_stats.losses += 1
            else:
                home_stats.draws += 1
        else:
            away_stats = self._away_stats[team]
            away_stats.matches_played += 1
            away_stats.goals_for += goals_for
            away_stats.goals_against += goals_against
            if goals_for > goals_against:
                away_stats.wins += 1
            elif goals_for < goals_against:
                away_stats.losses += 1
            else:
                away_stats.draws += 1
        
        # Son maç tarihini güncelle
        self._last_match_dates[team] = match_date
    
    def _update_h2h(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int
    ) -> None:
        """Head-to-Head istatistiklerini günceller"""
        key = (home_team, away_team)
        reverse_key = (away_team, home_team)
        
        if key not in self._h2h_stats:
            self._h2h_stats[key] = {'home_wins': 0, 'away_wins': 0, 'draws': 0}
        
        if reverse_key not in self._h2h_stats:
            self._h2h_stats[reverse_key] = {'home_wins': 0, 'away_wins': 0, 'draws': 0}
        
        if home_goals > away_goals:
            self._h2h_stats[key]['home_wins'] += 1
            self._h2h_stats[reverse_key]['away_wins'] += 1
        elif home_goals < away_goals:
            self._h2h_stats[key]['away_wins'] += 1
            self._h2h_stats[reverse_key]['home_wins'] += 1
        else:
            self._h2h_stats[key]['draws'] += 1
            self._h2h_stats[reverse_key]['draws'] += 1
    
    def _extract_features(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Maç için öznitelikleri çıkarır.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            match_date: Maç tarihi (dinlenme günü hesabı için)
            
        Returns:
            np.ndarray: Öznitelik vektörü
        """
        home_stats = self._team_stats.get(home_team, TeamStats())
        away_stats = self._team_stats.get(away_team, TeamStats())
        home_home = self._home_stats.get(home_team, TeamStats())
        away_away = self._away_stats.get(away_team, TeamStats())
        
        # Elo ratings
        home_elo = self._elo.get_rating(home_team)
        away_elo = self._elo.get_rating(away_team)
        
        # H2H
        h2h_key = (home_team, away_team)
        h2h = self._h2h_stats.get(h2h_key, {'home_wins': 0, 'away_wins': 0, 'draws': 0})
        
        # Dinlenme günleri
        if match_date:
            home_last = self._last_match_dates.get(home_team)
            away_last = self._last_match_dates.get(away_team)
            
            home_rest = (match_date - home_last).days if home_last else 7
            away_rest = (match_date - away_last).days if away_last else 7
        else:
            home_rest = 7
            away_rest = 7
        
        features = [
            home_elo,
            away_elo,
            home_elo - away_elo,
            home_stats.form_points,
            away_stats.form_points,
            home_stats.form_points - away_stats.form_points,
            home_stats.goals_per_game,
            away_stats.goals_per_game,
            home_stats.goals_against_per_game,
            away_stats.goals_against_per_game,
            home_stats.win_rate,
            away_stats.win_rate,
            h2h['home_wins'],
            h2h['away_wins'],
            h2h['draws'],
            home_home.win_rate if home_home.matches_played > 0 else 0.5,
            away_away.win_rate if away_away.matches_played > 0 else 0.3,
            min(home_rest, 14),  # Cap at 14 days
            min(away_rest, 14),
        ]
        
        return np.array(features)
    
    def _prepare_training_data(
        self,
        df: pd.DataFrame,
        home_col: str = 'home_team',
        away_col: str = 'away_team',
        home_goals_col: str = 'fthg',
        away_goals_col: str = 'ftag',
        result_col: str = 'result',
        date_col: str = 'date'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Eğitim verisi hazırlar.
        
        İstatistikleri kronolojik sırayla güncelleyerek
        her maç için o ana kadarki bilgileri kullanır (look-ahead bias yok).
        """
        logger.info("Eğitim verisi hazırlanıyor...")
        
        # Tarihe göre sırala
        df = df.copy()
        df['_date'] = pd.to_datetime(df[date_col])
        df = df.sort_values('_date')
        
        X_list = []
        y_list = []
        
        # Her maçı işle
        for idx, row in df.iterrows():
            home_team = row[home_col]
            away_team = row[away_col]
            home_goals = int(row[home_goals_col])
            away_goals = int(row[away_goals_col])
            result = row[result_col]
            match_date = row['_date']
            
            # Öznitelikleri çıkar (mevcut istatistiklerle)
            # Minimum 5 maç oynamamış takımlar için atla
            if (self._team_stats[home_team].matches_played >= 3 and 
                self._team_stats[away_team].matches_played >= 3):
                
                features = self._extract_features(home_team, away_team, match_date)
                X_list.append(features)
                y_list.append(result)
            
            # İstatistikleri güncelle (maç sonrası)
            self._update_team_stats(home_team, home_goals, away_goals, True, match_date)
            self._update_team_stats(away_team, away_goals, home_goals, False, match_date)
            self._update_h2h(home_team, away_team, home_goals, away_goals)
            self._elo.update(home_team, away_team, home_goals, away_goals)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        logger.info(f"Eğitim verisi hazır: {len(X)} örnek, {X.shape[1]} öznitelik")
        
        return X, y
    
    def fit(
        self,
        df: pd.DataFrame,
        home_col: str = 'home_team',
        away_col: str = 'away_team',
        home_goals_col: str = 'fthg',
        away_goals_col: str = 'ftag',
        result_col: str = 'result',
        date_col: str = 'date'
    ) -> "XGBoostPredictor":
        """
        Modeli eğitir.
        
        Args:
            df: Maç verileri DataFrame
            
        Returns:
            self: Eğitilmiş model
        """
        logger.info(f"XGBoost eğitimi başlıyor: {len(df)} maç")
        
        # İstatistikleri sıfırla
        self._team_stats.clear()
        self._home_stats.clear()
        self._away_stats.clear()
        self._h2h_stats.clear()
        self._elo = EloCalculator()
        self._last_match_dates.clear()
        
        # Eğitim verisi sakla
        self._training_df = df.copy()
        
        # Veri hazırla
        X, y = self._prepare_training_data(
            df, home_col, away_col, 
            home_goals_col, away_goals_col,
            result_col, date_col
        )
        
        # Label encode
        y_encoded = self._label_encoder.fit_transform(y)
        
        # XGBoost modelini oluştur
        self._base_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective='multi:softprob',
            num_class=3,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        logger.info("Model eğitiliyor...")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._base_model.fit(X, y_encoded)
        
        # Kalibrasyon
        if self.use_calibration and len(X) > 100:
            logger.info("Olasılık kalibrasyonu uygulanıyor...")
            try:
                self.model = CalibratedClassifierCV(
                    self._base_model, 
                    cv=3, 
                    method='isotonic'
                )
                self.model.fit(X, y_encoded)
            except Exception as e:
                logger.warning(f"Kalibrasyon başarısız: {e}. Base model kullanılıyor.")
                self.model = self._base_model
        else:
            self.model = self._base_model
        
        self.is_fitted = True
        
        # Cross-validation skoru
        try:
            cv_scores = cross_val_score(
                self._base_model, X, y_encoded, 
                cv=5, scoring='accuracy'
            )
            logger.info(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
        except Exception as e:
            logger.warning(f"CV hesaplanamadı: {e}")
        
        logger.info("XGBoost eğitimi tamamlandı")
        
        return self
    
    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Maç sonucu olasılıklarını tahmin eder.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            match_date: Maç tarihi
            
        Returns:
            Dict: {'home_win': float, 'draw': float, 'away_win': float}
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi. Önce fit() çağırın.")
        
        # Bilinmeyen takım kontrolü
        known_teams = set(self._team_stats.keys())
        
        if home_team not in known_teams:
            logger.warning(f"Bilinmeyen takım: {home_team}")
        if away_team not in known_teams:
            logger.warning(f"Bilinmeyen takım: {away_team}")
        
        # Öznitelikleri çıkar
        features = self._extract_features(home_team, away_team, match_date)
        X = features.reshape(1, -1)
        
        # Tahmin
        proba = self.model.predict_proba(X)[0]
        
        # Label mapping (A, D, H sıralaması)
        classes = self._label_encoder.classes_
        result = {}
        
        for i, cls in enumerate(classes):
            if cls == 'H':
                result['home_win'] = round(float(proba[i]), 4)
            elif cls == 'D':
                result['draw'] = round(float(proba[i]), 4)
            elif cls == 'A':
                result['away_win'] = round(float(proba[i]), 4)
        
        return result
    
    def predict(
        self,
        home_team: str,
        away_team: str,
        match_date: Optional[datetime] = None
    ) -> str:
        """
        En olası sonucu döndürür.
        
        Returns:
            str: 'H' (ev), 'D' (beraberlik), 'A' (deplasman)
        """
        proba = self.predict_proba(home_team, away_team, match_date)
        
        if proba['home_win'] >= proba['draw'] and proba['home_win'] >= proba['away_win']:
            return 'H'
        elif proba['away_win'] >= proba['draw']:
            return 'A'
        else:
            return 'D'
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Öznitelik önemlilik skorlarını döndürür.
        
        Returns:
            pd.DataFrame: Öznitelik isimleri ve önemlilikleri
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi.")
        
        # Base modelden importance al
        importance = self._base_model.feature_importances_
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        df = df.sort_values('importance', ascending=False)
        df = df.reset_index(drop=True)
        
        return df
    
    def get_team_elo_rankings(self) -> pd.DataFrame:
        """
        Takım Elo sıralamalarını döndürür.
        
        Returns:
            pd.DataFrame: Takım Elo tablosu
        """
        data = []
        for team, rating in self._elo.ratings.items():
            stats = self._team_stats.get(team, TeamStats())
            data.append({
                'team': team,
                'elo_rating': round(rating, 1),
                'matches': stats.matches_played,
                'wins': stats.wins,
                'draws': stats.draws,
                'losses': stats.losses,
                'gf': stats.goals_for,
                'ga': stats.goals_against,
                'gd': stats.goal_difference
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('elo_rating', ascending=False)
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        
        return df
    
    def save(self, path: Path) -> None:
        """Modeli dosyaya kaydeder."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'base_model': self._base_model,
            'label_encoder': self._label_encoder,
            'team_stats': dict(self._team_stats),
            'home_stats': dict(self._home_stats),
            'away_stats': dict(self._away_stats),
            'h2h_stats': self._h2h_stats,
            'elo_ratings': dict(self._elo.ratings),
            'last_match_dates': self._last_match_dates,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'min_child_weight': self.min_child_weight,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'use_calibration': self.use_calibration,
                'random_state': self.random_state
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"XGBoost modeli kaydedildi: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "XGBoostPredictor":
        """Modeli dosyadan yükler."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        params = model_data['params']
        predictor = cls(**params)
        
        predictor.model = model_data['model']
        predictor._base_model = model_data['base_model']
        predictor._label_encoder = model_data['label_encoder']
        predictor._team_stats = defaultdict(TeamStats, model_data['team_stats'])
        predictor._home_stats = defaultdict(TeamStats, model_data['home_stats'])
        predictor._away_stats = defaultdict(TeamStats, model_data['away_stats'])
        predictor._h2h_stats = model_data['h2h_stats']
        predictor._elo.ratings = defaultdict(lambda: 1500.0, model_data['elo_ratings'])
        predictor._last_match_dates = model_data['last_match_dates']
        predictor.feature_names = model_data['feature_names']
        predictor.is_fitted = model_data['is_fitted']
        
        logger.info(f"XGBoost modeli yüklendi: {path}")
        
        return predictor
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_teams = len(self._team_stats)
        return f"XGBoostPredictor({status}, teams={n_teams})"
