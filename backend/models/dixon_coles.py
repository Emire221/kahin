"""
The Oracle - Dixon-Coles Modeli

Bu modül, Dixon ve Coles (1997) tarafından geliştirilen düzeltilmiş
Poisson dağılımı modelini implement eder. Model, futbol maçlarının
skor olasılıklarını tahmin etmek için kullanılır.

Temel Özellikler:
    - Takım bazlı hücum ve savunma güçleri hesaplaması
    - Alt/Üst 2.5 gol olasılıkları
    - Karşılıklı Gol Var/Yok olasılıkları
    - İç saha avantajı faktörü
    - Düşük skorlu maçlar için düzeltme (rho parametresi)

Teorik Arka Plan:
    Standart Poisson modeli, gollerin bağımsız olduğunu varsayar.
    Ancak gerçekte 0-0, 1-0, 0-1, 1-1 gibi düşük skorlu maçlar
    beklenenden daha sık görülür. Dixon-Coles düzeltmesi (rho)
    bu sorunu çözer.

Kullanım:
    from backend.models.dixon_coles import DixonColesModel
    
    model = DixonColesModel()
    model.fit(matches_df)
    
    # Skor olasılıkları
    probs = model.predict_score_matrix("Arsenal", "Chelsea")
    
    # Maç sonucu olasılıkları
    result = model.predict_match_result("Arsenal", "Chelsea")
    # {'home_win': 0.45, 'draw': 0.28, 'away_win': 0.27}

Referans:
    Dixon, M. J., & Coles, S. G. (1997). Modelling association football 
    scores and inefficiencies in the football betting market. 
    Journal of the Royal Statistical Society: Series C, 46(2), 265-280.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import pickle
from pathlib import Path
import warnings

from loguru import logger


@dataclass
class TeamParameters:
    """Takım parametreleri"""
    attack: float = 1.0  # Hücum gücü
    defense: float = 1.0  # Savunma gücü (düşük = daha iyi savunma)


@dataclass
class DixonColesParameters:
    """Model parametreleri"""
    teams: Dict[str, TeamParameters] = field(default_factory=dict)
    home_advantage: float = 0.25  # İç saha avantajı (log scale)
    rho: float = -0.13  # Düzeltme faktörü (-0.2 ile 0 arası tipik)
    avg_goals: float = 2.75  # Lig gol ortalaması
    
    def to_dict(self) -> Dict[str, Any]:
        """Parametreleri dictionary'e çevirir"""
        return {
            'teams': {
                name: {'attack': tp.attack, 'defense': tp.defense}
                for name, tp in self.teams.items()
            },
            'home_advantage': self.home_advantage,
            'rho': self.rho,
            'avg_goals': self.avg_goals
        }


class DixonColesModel:
    """
    Dixon-Coles Skor Tahmin Modeli
    
    Bu model, Poisson dağılımını kullanarak futbol maçlarının
    olası skorlarını ve bunların olasılıklarını hesaplar.
    
    Attributes:
        params (DixonColesParameters): Model parametreleri
        is_fitted (bool): Model eğitilmiş mi?
        max_goals (int): Hesaplanacak maksimum gol sayısı
        
    Example:
        >>> model = DixonColesModel()
        >>> model.fit(df)
        >>> probs = model.predict_score_matrix("Arsenal", "Chelsea")
        >>> print(f"1-0 olasılığı: {probs[1, 0]:.2%}")
    """
    
    def __init__(
        self,
        max_goals: int = 10,
        time_decay: float = 0.0018,  # Zaman ağırlığı decay rate
        use_weights: bool = True
    ) -> None:
        """
        Dixon-Coles modelini başlatır.
        
        Args:
            max_goals: Hesaplanacak maksimum gol sayısı
            time_decay: Zaman bazlı ağırlık azalma oranı (günlük)
            use_weights: Yakın maçlara daha fazla ağırlık ver
        """
        self.max_goals = max_goals
        self.time_decay = time_decay
        self.use_weights = use_weights
        
        self.params = DixonColesParameters()
        self.is_fitted = False
        self._teams: List[str] = []
        self._training_date: Optional[datetime] = None
        
        logger.info("Dixon-Coles modeli başlatıldı")
    
    def _calculate_weights(
        self, 
        dates: pd.Series, 
        reference_date: Optional[datetime] = None
    ) -> np.ndarray:
        """
        Zaman bazlı ağırlıkları hesaplar.
        Yakın maçlar daha yüksek ağırlık alır.
        
        Args:
            dates: Maç tarihleri
            reference_date: Referans tarih (None ise son maç)
            
        Returns:
            np.ndarray: Ağırlıklar
        """
        if not self.use_weights:
            return np.ones(len(dates))
        
        # Tarihleri datetime'a çevir
        dates = pd.to_datetime(dates)
        
        if reference_date is None:
            reference_date = dates.max()
        
        # Gün farkını hesapla
        days_diff = (reference_date - dates).dt.days.values
        
        # Exponential decay
        weights = np.exp(-self.time_decay * days_diff)
        
        return weights
    
    def _rho_correction(
        self, 
        home_goals: int, 
        away_goals: int, 
        lambda_home: float, 
        lambda_away: float, 
        rho: float
    ) -> float:
        """
        Dixon-Coles düzeltme faktörünü hesaplar.
        
        Düşük skorlu maçlar (0-0, 1-0, 0-1, 1-1) için
        Poisson dağılımının yetersiz kaldığı durumları düzeltir.
        
        Args:
            home_goals: Ev sahibi gol sayısı
            away_goals: Deplasman gol sayısı
            lambda_home: Ev sahibi beklenen gol
            lambda_away: Deplasman beklenen gol
            rho: Düzeltme parametresi
            
        Returns:
            float: Düzeltme çarpanı
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - lambda_home * lambda_away * rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + lambda_home * rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + lambda_away * rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - rho
        else:
            return 1.0
    
    def _poisson_probability(
        self, 
        goals: int, 
        lambda_val: float
    ) -> float:
        """Poisson olasılığını hesaplar"""
        return poisson.pmf(goals, lambda_val)
    
    def _calculate_expected_goals(
        self, 
        home_team: str, 
        away_team: str
    ) -> Tuple[float, float]:
        """
        Beklenen gol sayılarını hesaplar.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            
        Returns:
            Tuple[float, float]: (ev beklenen gol, dep beklenen gol)
        """
        if home_team not in self.params.teams:
            logger.warning(f"Bilinmeyen takım: {home_team}, ortalama değerler kullanılıyor")
            home_params = TeamParameters()
        else:
            home_params = self.params.teams[home_team]
        
        if away_team not in self.params.teams:
            logger.warning(f"Bilinmeyen takım: {away_team}, ortalama değerler kullanılıyor")
            away_params = TeamParameters()
        else:
            away_params = self.params.teams[away_team]
        
        # Beklenen goller
        lambda_home = (
            home_params.attack * 
            away_params.defense * 
            np.exp(self.params.home_advantage)
        )
        
        lambda_away = (
            away_params.attack * 
            home_params.defense
        )
        
        return lambda_home, lambda_away
    
    def _negative_log_likelihood(
        self, 
        params_vector: np.ndarray,
        home_teams: np.ndarray,
        away_teams: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray,
        weights: np.ndarray,
        team_indices: Dict[str, int]
    ) -> float:
        """
        Negatif log-likelihood hesaplar (minimize edilecek).
        
        Bu fonksiyon, modelin veriyle ne kadar uyumlu olduğunu ölçer.
        Daha düşük değer = daha iyi uyum.
        """
        n_teams = len(team_indices)
        
        # Parametreleri ayrıştır
        attack = params_vector[:n_teams]
        defense = params_vector[n_teams:2*n_teams]
        home_adv = params_vector[2*n_teams]
        rho = params_vector[2*n_teams + 1]
        
        # Constraint: Toplam attack = n_teams (normalizasyon)
        attack = attack / np.mean(attack)
        defense = defense / np.mean(defense)
        
        total_nll = 0.0
        
        for i in range(len(home_teams)):
            h_idx = team_indices[home_teams[i]]
            a_idx = team_indices[away_teams[i]]
            
            lambda_h = attack[h_idx] * defense[a_idx] * np.exp(home_adv)
            lambda_a = attack[a_idx] * defense[h_idx]
            
            # Poisson olasılıkları
            p_home = poisson.pmf(home_goals[i], lambda_h)
            p_away = poisson.pmf(away_goals[i], lambda_a)
            
            # Dixon-Coles düzeltmesi
            rho_corr = self._rho_correction(
                home_goals[i], away_goals[i],
                lambda_h, lambda_a, rho
            )
            
            prob = p_home * p_away * rho_corr
            
            # Log olasılık (0'a çok yakın değerleri koru)
            if prob > 1e-10:
                total_nll -= weights[i] * np.log(prob)
            else:
                total_nll += weights[i] * 20  # Ceza
        
        return total_nll
    
    def fit(
        self, 
        df: pd.DataFrame,
        home_col: str = 'home_team',
        away_col: str = 'away_team',
        home_goals_col: str = 'fthg',
        away_goals_col: str = 'ftag',
        date_col: str = 'date'
    ) -> "DixonColesModel":
        """
        Modeli eğitir.
        
        Args:
            df: Maç verileri DataFrame
            home_col: Ev sahibi takım sütunu
            away_col: Deplasman takım sütunu
            home_goals_col: Ev sahibi gol sütunu
            away_goals_col: Deplasman gol sütunu
            date_col: Tarih sütunu
            
        Returns:
            self: Eğitilmiş model
            
        Raises:
            ValueError: Gerekli sütunlar eksikse
        """
        logger.info(f"Dixon-Coles eğitimi başlıyor: {len(df)} maç")
        
        # Gerekli sütunları kontrol et
        required = [home_col, away_col, home_goals_col, away_goals_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Eksik sütunlar: {missing}")
        
        # Veriyi hazırla
        df = df.copy()
        df = df.dropna(subset=required)
        
        # Takım listesi
        self._teams = sorted(set(df[home_col].unique()) | set(df[away_col].unique()))
        team_indices = {team: i for i, team in enumerate(self._teams)}
        n_teams = len(self._teams)
        
        logger.info(f"Toplam takım sayısı: {n_teams}")
        
        # Ağırlıkları hesapla
        if date_col in df.columns:
            weights = self._calculate_weights(df[date_col])
            self._training_date = pd.to_datetime(df[date_col]).max()
        else:
            weights = np.ones(len(df))
            self._training_date = datetime.now()
        
        # Numpy dizileri
        home_teams = df[home_col].values
        away_teams = df[away_col].values
        home_goals = df[home_goals_col].values.astype(int)
        away_goals = df[away_goals_col].values.astype(int)
        
        # Başlangıç değerleri
        initial_attack = np.ones(n_teams)
        initial_defense = np.ones(n_teams)
        initial_home_adv = 0.25
        initial_rho = -0.1
        
        x0 = np.concatenate([
            initial_attack,
            initial_defense,
            [initial_home_adv, initial_rho]
        ])
        
        # Parametrelerin sınırları
        bounds = (
            [(0.2, 3.0)] * n_teams +  # Attack
            [(0.2, 3.0)] * n_teams +  # Defense
            [(0.0, 0.5)] +            # Home advantage
            [(-0.3, 0.1)]             # Rho
        )
        
        logger.info("Optimizasyon başlıyor...")
        
        # Optimizasyon
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            result = minimize(
                self._negative_log_likelihood,
                x0,
                args=(
                    home_teams, away_teams,
                    home_goals, away_goals,
                    weights, team_indices
                ),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'disp': False}
            )
        
        if not result.success:
            logger.warning(f"Optimizasyon tam yakınsamadı: {result.message}")
        
        # Parametreleri kaydet
        attack = result.x[:n_teams]
        defense = result.x[n_teams:2*n_teams]
        
        # Normalizasyon
        attack = attack / np.mean(attack)
        defense = defense / np.mean(defense)
        
        self.params = DixonColesParameters(
            teams={
                team: TeamParameters(attack=attack[i], defense=defense[i])
                for team, i in team_indices.items()
            },
            home_advantage=result.x[2*n_teams],
            rho=result.x[2*n_teams + 1],
            avg_goals=np.mean(home_goals) + np.mean(away_goals)
        )
        
        self.is_fitted = True
        
        logger.info(
            f"Eğitim tamamlandı. "
            f"Home Adv: {self.params.home_advantage:.3f}, "
            f"Rho: {self.params.rho:.3f}"
        )
        
        return self
    
    def predict_score_matrix(
        self, 
        home_team: str, 
        away_team: str
    ) -> np.ndarray:
        """
        Skor olasılık matrisini hesaplar.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            
        Returns:
            np.ndarray: (max_goals+1, max_goals+1) boyutunda olasılık matrisi
                       matrix[i, j] = i-j skorunun olasılığı
                       
        Example:
            >>> probs = model.predict_score_matrix("Arsenal", "Chelsea")
            >>> print(f"0-0: {probs[0,0]:.2%}")
            >>> print(f"2-1: {probs[2,1]:.2%}")
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi. Önce fit() çağırın.")
        
        lambda_home, lambda_away = self._calculate_expected_goals(home_team, away_team)
        
        # Olasılık matrisi
        matrix = np.zeros((self.max_goals + 1, self.max_goals + 1))
        
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p_home = self._poisson_probability(i, lambda_home)
                p_away = self._poisson_probability(j, lambda_away)
                rho_corr = self._rho_correction(i, j, lambda_home, lambda_away, self.params.rho)
                
                matrix[i, j] = p_home * p_away * rho_corr
        
        # Normalize (toplam 1 olsun)
        matrix = matrix / matrix.sum()
        
        return matrix
    
    def predict_match_result(
        self, 
        home_team: str, 
        away_team: str
    ) -> Dict[str, float]:
        """
        Maç sonucu olasılıklarını hesaplar.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            
        Returns:
            Dict: {'home_win': float, 'draw': float, 'away_win': float}
        """
        matrix = self.predict_score_matrix(home_team, away_team)
        
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                if i > j:
                    home_win += matrix[i, j]
                elif i == j:
                    draw += matrix[i, j]
                else:
                    away_win += matrix[i, j]
        
        return {
            'home_win': round(home_win, 4),
            'draw': round(draw, 4),
            'away_win': round(away_win, 4)
        }
    
    def predict_expected_goals(
        self, 
        home_team: str, 
        away_team: str
    ) -> Dict[str, float]:
        """
        Beklenen gol sayılarını döndürür.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            
        Returns:
            Dict: {'home': float, 'away': float, 'total': float}
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi.")
        
        lambda_home, lambda_away = self._calculate_expected_goals(home_team, away_team)
        
        return {
            'home': round(lambda_home, 2),
            'away': round(lambda_away, 2),
            'total': round(lambda_home + lambda_away, 2)
        }
    
    def predict_over_under(
        self, 
        home_team: str, 
        away_team: str,
        threshold: float = 2.5
    ) -> Dict[str, float]:
        """
        Alt/Üst olasılıklarını hesaplar.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            threshold: Gol eşiği (2.5, 3.5 vb.)
            
        Returns:
            Dict: {'over': float, 'under': float}
        """
        matrix = self.predict_score_matrix(home_team, away_team)
        
        under = 0.0
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                if i + j < threshold:
                    under += matrix[i, j]
        
        over = 1 - under
        
        return {
            'over': round(over, 4),
            'under': round(under, 4),
            'threshold': threshold
        }
    
    def predict_btts(
        self, 
        home_team: str, 
        away_team: str
    ) -> Dict[str, float]:
        """
        Karşılıklı Gol Var/Yok olasılıklarını hesaplar.
        BTTS = Both Teams To Score
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            
        Returns:
            Dict: {'yes': float, 'no': float}
        """
        matrix = self.predict_score_matrix(home_team, away_team)
        
        btts_no = 0.0
        
        # İlk satır (ev sahibi 0 gol) veya ilk sütun (deplasman 0 gol)
        for j in range(self.max_goals + 1):
            btts_no += matrix[0, j]  # Ev sahibi 0 gol attı
        
        for i in range(1, self.max_goals + 1):
            btts_no += matrix[i, 0]  # Deplasman 0 gol attı (zaten 0-0 saydık)
        
        btts_yes = 1 - btts_no
        
        return {
            'yes': round(btts_yes, 4),
            'no': round(btts_no, 4)
        }
    
    def predict_correct_score(
        self, 
        home_team: str, 
        away_team: str,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        En olası skorları döndürür.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            top_n: Kaç skor döndürülsün
            
        Returns:
            List[Dict]: [{'score': '1-1', 'probability': 0.12}, ...]
        """
        matrix = self.predict_score_matrix(home_team, away_team)
        
        scores = []
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                scores.append({
                    'home_goals': i,
                    'away_goals': j,
                    'score': f"{i}-{j}",
                    'probability': round(matrix[i, j], 4)
                })
        
        # Olasılığa göre sırala
        scores.sort(key=lambda x: x['probability'], reverse=True)
        
        return scores[:top_n]
    
    def get_team_strengths(self) -> pd.DataFrame:
        """
        Tüm takımların hücum ve savunma güçlerini döndürür.
        
        Returns:
            pd.DataFrame: Takım güçleri tablosu
        """
        if not self.is_fitted:
            raise ValueError("Model henüz eğitilmedi.")
        
        data = []
        for team, params in self.params.teams.items():
            data.append({
                'team': team,
                'attack': round(params.attack, 3),
                'defense': round(params.defense, 3),
                'overall': round(params.attack / params.defense, 3)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('overall', ascending=False)
        df = df.reset_index(drop=True)
        df.index = df.index + 1  # 1'den başlat
        
        return df
    
    def save(self, path: Path) -> None:
        """
        Modeli dosyaya kaydeder.
        
        Args:
            path: Kayıt yolu (.pkl)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'params': self.params.to_dict(),
            'teams': self._teams,
            'training_date': self._training_date,
            'max_goals': self.max_goals,
            'time_decay': self.time_decay,
            'use_weights': self.use_weights,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model kaydedildi: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "DixonColesModel":
        """
        Modeli dosyadan yükler.
        
        Args:
            path: Model dosyası yolu (.pkl)
            
        Returns:
            DixonColesModel: Yüklenen model
        """
        path = Path(path)
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            max_goals=model_data['max_goals'],
            time_decay=model_data['time_decay'],
            use_weights=model_data['use_weights']
        )
        
        # Parametreleri yükle
        params_dict = model_data['params']
        model.params = DixonColesParameters(
            teams={
                name: TeamParameters(attack=tp['attack'], defense=tp['defense'])
                for name, tp in params_dict['teams'].items()
            },
            home_advantage=params_dict['home_advantage'],
            rho=params_dict['rho'],
            avg_goals=params_dict['avg_goals']
        )
        
        model._teams = model_data['teams']
        model._training_date = model_data['training_date']
        model.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model yüklendi: {path}")
        
        return model
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        n_teams = len(self._teams) if self._teams else 0
        return f"DixonColesModel({status}, teams={n_teams})"
