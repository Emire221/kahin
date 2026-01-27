"""
The Oracle - Value Bet Hesaplayıcı

Bu modül, bahis oranlarındaki "değer"i (value) tespit eder.
Model olasılığı ile bahis oranı arasındaki farkı analiz eder.

Value Bet Mantığı:
    - Bahisçi bir maç için 2.00 oran verirse, %50 olasılık ima eder
    - Modelimiz %55 olasılık hesaplarsa, bu bir "value bet"tir
    - EV = 0.55 × 2.00 = 1.10 (> 1.05 eşik değeri)

Kullanım:
    from backend.services.value_calculator import ValueCalculator
    
    calc = ValueCalculator(threshold=1.05)
    
    # Value bet kontrolü
    is_value = calc.is_value_bet(probability=0.55, odds=2.00)
    
    # Detaylı analiz
    analysis = calc.analyze_bet(
        probability=0.55,
        odds=2.00,
        home_team="Arsenal",
        away_team="Chelsea"
    )
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from loguru import logger


class BetType(Enum):
    """Bahis tipleri"""
    HOME_WIN = "MS1"      # Maç Sonucu: Ev sahibi
    DRAW = "MS0"          # Maç Sonucu: Beraberlik
    AWAY_WIN = "MS2"      # Maç Sonucu: Deplasman
    OVER_25 = "UST25"     # Üst 2.5 gol
    UNDER_25 = "ALT25"    # Alt 2.5 gol
    BTTS_YES = "KG_VAR"   # Karşılıklı Gol Var
    BTTS_NO = "KG_YOK"    # Karşılıklı Gol Yok


@dataclass
class ValueBetResult:
    """Value bet analiz sonucu"""
    home_team: str
    away_team: str
    bet_type: str
    probability: float      # Model olasılığı
    implied_prob: float     # Oranın ima ettiği olasılık
    odds: float             # Bahis oranı
    expected_value: float   # EV = prob × odds
    edge: float             # Avantaj = model prob - implied prob
    is_value: bool          # Value bet mi?
    confidence: str         # LOW, MEDIUM, HIGH
    recommendation: str     # Tavsiye
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ValueCalculator:
    """
    Value Bet Hesaplayıcı
    
    Bahis oranlarındaki değeri analiz eder ve value bet'leri tespit eder.
    
    Attributes:
        threshold (float): Minimum EV eşik değeri (varsayılan 1.05)
        min_odds (float): Minimum kabul edilebilir oran
        max_odds (float): Maksimum kabul edilebilir oran
        
    Example:
        >>> calc = ValueCalculator(threshold=1.05)
        >>> result = calc.is_value_bet(0.55, 2.00)
        >>> print(result)  # True
    """
    
    def __init__(
        self,
        threshold: float = 1.05,
        min_odds: float = 1.20,
        max_odds: float = 10.0,
        min_edge: float = 0.03  # Minimum %3 edge
    ) -> None:
        """
        ValueCalculator'ı başlatır.
        
        Args:
            threshold: Minimum EV eşik değeri
            min_odds: Minimum kabul edilebilir oran
            max_odds: Maksimum kabul edilebilir oran
            min_edge: Minimum avantaj yüzdesi
        """
        self.threshold = threshold
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.min_edge = min_edge
        
        logger.info(f"ValueCalculator başlatıldı: EV Eşik={threshold}, Min Edge={min_edge*100}%")
    
    def odds_to_probability(self, odds: float) -> float:
        """
        Bahis oranını olasılığa çevirir.
        
        Args:
            odds: Decimal bahis oranı
            
        Returns:
            float: İma edilen olasılık (0-1 arası)
        """
        if odds <= 0:
            return 0.0
        return 1.0 / odds
    
    def probability_to_odds(self, probability: float) -> float:
        """
        Olasılığı bahis oranına çevirir.
        
        Args:
            probability: Olasılık (0-1 arası)
            
        Returns:
            float: Decimal bahis oranı
        """
        if probability <= 0:
            return 0.0
        return 1.0 / probability
    
    def calculate_ev(self, probability: float, odds: float) -> float:
        """
        Expected Value (Beklenen Değer) hesaplar.
        
        EV = Olasılık × Oran
        
        EV > 1: Uzun vadede karlı bahis
        EV = 1: Nötr bahis  
        EV < 1: Uzun vadede zararlı bahis
        
        Args:
            probability: Model olasılığı
            odds: Bahis oranı
            
        Returns:
            float: Expected Value
        """
        return probability * odds
    
    def calculate_edge(self, probability: float, odds: float) -> float:
        """
        Edge (avantaj) hesaplar.
        
        Edge = Model Olasılığı - Bahisçinin İma Ettiği Olasılık
        
        Args:
            probability: Model olasılığı
            odds: Bahis oranı
            
        Returns:
            float: Edge (pozitif = avantaj, negatif = dezavantaj)
        """
        implied_prob = self.odds_to_probability(odds)
        return probability - implied_prob
    
    def calculate_kelly_stake(
        self, 
        probability: float, 
        odds: float,
        fraction: float = 0.25  # Quarter Kelly (daha güvenli)
    ) -> float:
        """
        Kelly Criterion ile optimal bahis miktarı hesaplar.
        
        Full Kelly = (p × b - q) / b
        Burada:
            p = kazanma olasılığı
            q = kaybetme olasılığı (1 - p)
            b = net oran (odds - 1)
        
        Args:
            probability: Kazanma olasılığı
            odds: Bahis oranı
            fraction: Kelly fraksiyonu (0.25 = Quarter Kelly)
            
        Returns:
            float: Bankroll'un yüzde kaçıyla bahis yapılmalı
        """
        if odds <= 1 or probability <= 0 or probability >= 1:
            return 0.0
        
        b = odds - 1  # Net oran
        p = probability
        q = 1 - p
        
        kelly = (p * b - q) / b
        
        # Negatif Kelly = bahis yapma
        if kelly <= 0:
            return 0.0
        
        # Fractional Kelly (daha güvenli)
        return kelly * fraction
    
    def is_value_bet(self, probability: float, odds: float) -> bool:
        """
        Value bet olup olmadığını kontrol eder.
        
        Args:
            probability: Model olasılığı
            odds: Bahis oranı
            
        Returns:
            bool: Value bet ise True
        """
        # Oran sınırları
        if odds < self.min_odds or odds > self.max_odds:
            return False
        
        # EV kontrolü
        ev = self.calculate_ev(probability, odds)
        if ev < self.threshold:
            return False
        
        # Edge kontrolü
        edge = self.calculate_edge(probability, odds)
        if edge < self.min_edge:
            return False
        
        return True
    
    def get_confidence_level(self, probability: float, edge: float) -> str:
        """
        Güven seviyesini belirler.
        
        Args:
            probability: Model olasılığı
            edge: Avantaj
            
        Returns:
            str: LOW, MEDIUM, HIGH
        """
        # Yüksek olasılık + yüksek edge = yüksek güven
        if probability >= 0.50 and edge >= 0.10:
            return "HIGH"
        elif probability >= 0.40 and edge >= 0.05:
            return "MEDIUM"
        else:
            return "LOW"
    
    def analyze_bet(
        self,
        probability: float,
        odds: float,
        bet_type: str = "MS1",
        home_team: str = "",
        away_team: str = ""
    ) -> ValueBetResult:
        """
        Detaylı value bet analizi yapar.
        
        Args:
            probability: Model olasılığı
            odds: Bahis oranı
            bet_type: Bahis tipi
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            
        Returns:
            ValueBetResult: Analiz sonucu
        """
        ev = self.calculate_ev(probability, odds)
        edge = self.calculate_edge(probability, odds)
        implied_prob = self.odds_to_probability(odds)
        is_value = self.is_value_bet(probability, odds)
        confidence = self.get_confidence_level(probability, edge)
        
        # Tavsiye oluştur
        if is_value:
            if confidence == "HIGH":
                recommendation = "🟢 GÜÇLÜ VALUE - Bahis yapılabilir"
            elif confidence == "MEDIUM":
                recommendation = "🟡 ORTA VALUE - Dikkatli değerlendirin"
            else:
                recommendation = "🟠 DÜŞÜK VALUE - Riskli olabilir"
        else:
            recommendation = "🔴 VALUE YOK - Bahis önerilmez"
        
        return ValueBetResult(
            home_team=home_team,
            away_team=away_team,
            bet_type=bet_type,
            probability=round(probability, 4),
            implied_prob=round(implied_prob, 4),
            odds=odds,
            expected_value=round(ev, 4),
            edge=round(edge, 4),
            is_value=is_value,
            confidence=confidence,
            recommendation=recommendation
        )
    
    def analyze_match(
        self,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_odds: float,
        draw_odds: float,
        away_odds: float,
        home_team: str = "",
        away_team: str = "",
        over_25_prob: Optional[float] = None,
        over_25_odds: Optional[float] = None,
        under_25_odds: Optional[float] = None,
        btts_yes_prob: Optional[float] = None,
        btts_yes_odds: Optional[float] = None,
        btts_no_odds: Optional[float] = None
    ) -> Dict[str, ValueBetResult]:
        """
        Bir maçın tüm bahis tiplerini analiz eder.
        
        Returns:
            Dict[str, ValueBetResult]: Bahis tipi -> Analiz sonucu
        """
        results = {}
        
        # Maç Sonucu bahisleri
        results['MS1'] = self.analyze_bet(
            home_win_prob, home_odds, "MS1", home_team, away_team
        )
        results['MS0'] = self.analyze_bet(
            draw_prob, draw_odds, "MS0", home_team, away_team
        )
        results['MS2'] = self.analyze_bet(
            away_win_prob, away_odds, "MS2", home_team, away_team
        )
        
        # Alt/Üst 2.5 bahisleri
        if over_25_prob is not None and over_25_odds is not None:
            results['UST25'] = self.analyze_bet(
                over_25_prob, over_25_odds, "UST25", home_team, away_team
            )
        
        if over_25_prob is not None and under_25_odds is not None:
            under_25_prob = 1 - over_25_prob
            results['ALT25'] = self.analyze_bet(
                under_25_prob, under_25_odds, "ALT25", home_team, away_team
            )
        
        # KG Var/Yok bahisleri
        if btts_yes_prob is not None and btts_yes_odds is not None:
            results['KG_VAR'] = self.analyze_bet(
                btts_yes_prob, btts_yes_odds, "KG_VAR", home_team, away_team
            )
        
        if btts_yes_prob is not None and btts_no_odds is not None:
            btts_no_prob = 1 - btts_yes_prob
            results['KG_YOK'] = self.analyze_bet(
                btts_no_prob, btts_no_odds, "KG_YOK", home_team, away_team
            )
        
        return results
    
    def find_value_bets(
        self,
        analysis_results: Dict[str, ValueBetResult]
    ) -> List[ValueBetResult]:
        """
        Value bet'leri filtreler.
        
        Args:
            analysis_results: analyze_match sonuçları
            
        Returns:
            List[ValueBetResult]: Sadece value bet olan sonuçlar
        """
        value_bets = []
        
        for bet_type, result in analysis_results.items():
            if result.is_value:
                value_bets.append(result)
        
        # EV'ye göre sırala (en yüksek önce)
        value_bets.sort(key=lambda x: x.expected_value, reverse=True)
        
        return value_bets
    
    def get_best_bet(
        self,
        analysis_results: Dict[str, ValueBetResult]
    ) -> Optional[ValueBetResult]:
        """
        En iyi value bet'i döndürür.
        
        Args:
            analysis_results: analyze_match sonuçları
            
        Returns:
            ValueBetResult: En iyi value bet veya None
        """
        value_bets = self.find_value_bets(analysis_results)
        
        if not value_bets:
            return None
        
        return value_bets[0]  # En yüksek EV


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def calculate_roi(total_pnl: float, total_staked: float) -> float:
    """
    ROI (Return on Investment) hesaplar.
    
    ROI = (Toplam Kar/Zarar / Toplam Yatırılan) × 100
    
    Args:
        total_pnl: Toplam kar/zarar
        total_staked: Toplam yatırılan miktar
        
    Returns:
        float: ROI yüzdesi
    """
    if total_staked <= 0:
        return 0.0
    return (total_pnl / total_staked) * 100


def calculate_hit_rate(wins: int, total_bets: int) -> float:
    """
    Hit Rate (isabet oranı) hesaplar.
    
    Args:
        wins: Kazanılan bahis sayısı
        total_bets: Toplam bahis sayısı
        
    Returns:
        float: Hit rate yüzdesi
    """
    if total_bets <= 0:
        return 0.0
    return (wins / total_bets) * 100


def calculate_max_drawdown(balance_history: List[float]) -> float:
    """
    Maximum Drawdown hesaplar.
    
    Max Drawdown = En yüksek noktadan en düşük noktaya düşüş yüzdesi
    
    Args:
        balance_history: Bakiye geçmişi listesi
        
    Returns:
        float: Max drawdown yüzdesi
    """
    if not balance_history:
        return 0.0
    
    peak = balance_history[0]
    max_dd = 0.0
    
    for balance in balance_history:
        if balance > peak:
            peak = balance
        
        if peak > 0:
            dd = (peak - balance) / peak * 100
            if dd > max_dd:
                max_dd = dd
    
    return max_dd


def calculate_sharpe_ratio(
    returns: List[float],
    risk_free_rate: float = 0.0
) -> float:
    """
    Sharpe Ratio hesaplar.
    
    Risk ayarlı getiri ölçüsü. Yüksek = daha iyi risk/getiri oranı.
    
    Args:
        returns: Getiri listesi
        risk_free_rate: Risksiz faiz oranı
        
    Returns:
        float: Sharpe Ratio
    """
    if len(returns) < 2:
        return 0.0
    
    import numpy as np
    
    returns_array = np.array(returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    
    if std_return == 0:
        return 0.0
    
    return (mean_return - risk_free_rate) / std_return
