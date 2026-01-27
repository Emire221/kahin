"""
The Oracle - Sanal Kasa (Wallet) Modülü

Bu modül, backtest sırasında sanal bahis işlemlerini yönetir.
Kar/zarar takibi, drawdown hesaplaması ve işlem geçmişi kaydı yapar.

Kullanım:
    from backend.simulation.wallet import Wallet
    
    wallet = Wallet(initial_balance=1000.0, stake=10.0)
    
    # Bahis yap
    wallet.place_bet(
        match_id=1,
        bet_type="MS1",
        odds=2.10,
        won=True
    )
    
    # Özet al
    summary = wallet.get_summary()
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path

from loguru import logger


class BetResult(Enum):
    """Bahis sonucu"""
    WIN = "win"
    LOSE = "lose"
    VOID = "void"  # İptal edilmiş bahis


@dataclass
class Transaction:
    """Tek bir bahis işlemi"""
    id: int
    date: str
    match_id: int
    home_team: str
    away_team: str
    bet_type: str  # MS1, MS0, MS2, ALT25, UST25, KG_VAR, KG_YOK
    odds: float
    stake: float
    predicted_prob: float  # Model olasılığı
    expected_value: float  # EV = prob * odds
    result: Optional[BetResult] = None
    pnl: float = 0.0  # Profit/Loss
    balance_after: float = 0.0
    actual_result: Optional[str] = None  # Gerçek maç sonucu
    
    def to_dict(self) -> Dict:
        """Dictionary'e çevirir"""
        d = asdict(self)
        d['result'] = self.result.value if self.result else None
        return d


@dataclass
class WalletStats:
    """Kasa istatistikleri"""
    initial_balance: float = 0.0
    current_balance: float = 0.0
    total_bets: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_staked: float = 0.0
    total_returns: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0  # Return on Investment (%)
    hit_rate: float = 0.0  # Kazanma oranı (%)
    peak_balance: float = 0.0  # En yüksek bakiye
    lowest_balance: float = 0.0  # En düşük bakiye
    max_drawdown: float = 0.0  # En büyük düşüş (%)
    avg_odds: float = 0.0  # Ortalama oran
    avg_ev: float = 0.0  # Ortalama EV
    
    def to_dict(self) -> Dict:
        return asdict(self)


class Wallet:
    """
    Sanal Kasa Yöneticisi
    
    Backtest sırasında bahis işlemlerini simüle eder,
    kar/zarar takibi yapar ve performans metrikleri hesaplar.
    
    Attributes:
        balance (float): Mevcut bakiye
        initial_balance (float): Başlangıç bakiyesi
        stake (float): Sabit bahis miktarı
        value_threshold (float): Value bet eşik değeri
        
    Example:
        >>> wallet = Wallet(initial_balance=1000, stake=10)
        >>> wallet.place_bet(1, "MS1", 2.5, True, 0.45, "Arsenal", "Chelsea")
        >>> print(wallet.balance)  # 1015.0
        >>> print(wallet.get_summary()['roi'])  # 1.5
    """
    
    def __init__(
        self,
        initial_balance: float = 1000.0,
        stake: float = 10.0,
        value_threshold: float = 1.05
    ) -> None:
        """
        Wallet'ı başlatır.
        
        Args:
            initial_balance: Başlangıç bakiyesi
            stake: Sabit bahis miktarı (Flat Stake)
            value_threshold: Minimum EV eşiği (varsayılan 1.05)
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.stake = stake
        self.value_threshold = value_threshold
        
        self._transactions: List[Transaction] = []
        self._transaction_counter = 0
        
        # İstatistik takibi
        self._peak_balance = initial_balance
        self._lowest_balance = initial_balance
        self._total_staked = 0.0
        self._total_returns = 0.0
        self._wins = 0
        self._losses = 0
        
        logger.info(
            f"Wallet başlatıldı: "
            f"Bakiye={initial_balance}, Stake={stake}, EV Eşik={value_threshold}"
        )
    
    def reset(self) -> None:
        """Kasayı sıfırlar"""
        self.balance = self.initial_balance
        self._transactions.clear()
        self._transaction_counter = 0
        self._peak_balance = self.initial_balance
        self._lowest_balance = self.initial_balance
        self._total_staked = 0.0
        self._total_returns = 0.0
        self._wins = 0
        self._losses = 0
        
        logger.info("Wallet sıfırlandı")
    
    def can_bet(self) -> bool:
        """Bahis yapılabilir mi kontrol eder"""
        return self.balance >= self.stake
    
    def is_value_bet(self, probability: float, odds: float) -> bool:
        """
        Value bet olup olmadığını kontrol eder.
        
        Value = Probability × Odds > value_threshold
        
        Args:
            probability: Model olasılığı (0-1 arası)
            odds: Bahis oranı
            
        Returns:
            bool: Value bet ise True
        """
        expected_value = probability * odds
        return expected_value >= self.value_threshold
    
    def calculate_ev(self, probability: float, odds: float) -> float:
        """Expected Value hesaplar"""
        return probability * odds
    
    def place_bet(
        self,
        match_id: int,
        bet_type: str,
        odds: float,
        won: bool,
        predicted_prob: float,
        home_team: str = "",
        away_team: str = "",
        date: str = "",
        actual_result: str = ""
    ) -> Transaction:
        """
        Bahis yapar ve işlemi kaydeder.
        
        Args:
            match_id: Maç ID'si
            bet_type: Bahis tipi (MS1, MS0, MS2, vb.)
            odds: Bahis oranı
            won: Bahis kazandı mı
            predicted_prob: Model olasılığı
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            date: Maç tarihi
            actual_result: Gerçek sonuç
            
        Returns:
            Transaction: İşlem kaydı
        """
        if not self.can_bet():
            logger.warning(f"Yetersiz bakiye! Bakiye: {self.balance}, Stake: {self.stake}")
            return None
        
        self._transaction_counter += 1
        
        # EV hesapla
        ev = self.calculate_ev(predicted_prob, odds)
        
        # Bahisi yap
        self.balance -= self.stake
        self._total_staked += self.stake
        
        # Sonucu işle
        if won:
            returns = self.stake * odds
            pnl = returns - self.stake
            self.balance += returns
            self._total_returns += returns
            self._wins += 1
            result = BetResult.WIN
        else:
            pnl = -self.stake
            self._losses += 1
            result = BetResult.LOSE
        
        # Peak ve lowest güncelle
        if self.balance > self._peak_balance:
            self._peak_balance = self.balance
        if self.balance < self._lowest_balance:
            self._lowest_balance = self.balance
        
        # Transaction oluştur
        transaction = Transaction(
            id=self._transaction_counter,
            date=date or datetime.now().strftime("%Y-%m-%d"),
            match_id=match_id,
            home_team=home_team,
            away_team=away_team,
            bet_type=bet_type,
            odds=odds,
            stake=self.stake,
            predicted_prob=predicted_prob,
            expected_value=ev,
            result=result,
            pnl=pnl,
            balance_after=self.balance,
            actual_result=actual_result
        )
        
        self._transactions.append(transaction)
        
        logger.debug(
            f"Bahis #{self._transaction_counter}: "
            f"{home_team} vs {away_team}, {bet_type}@{odds:.2f}, "
            f"{'✓ WIN' if won else '✗ LOSE'}, PnL: {pnl:+.2f}, "
            f"Bakiye: {self.balance:.2f}"
        )
        
        return transaction
    
    def get_summary(self) -> WalletStats:
        """
        Kasa özet istatistiklerini döndürür.
        
        Returns:
            WalletStats: İstatistikler
        """
        total_bets = len(self._transactions)
        
        if total_bets == 0:
            return WalletStats(
                initial_balance=self.initial_balance,
                current_balance=self.balance
            )
        
        # ROI hesapla
        total_pnl = self.balance - self.initial_balance
        roi = (total_pnl / self._total_staked * 100) if self._total_staked > 0 else 0
        
        # Hit rate
        hit_rate = (self._wins / total_bets * 100) if total_bets > 0 else 0
        
        # Max drawdown
        max_drawdown = 0
        if self._peak_balance > 0:
            max_drawdown = ((self._peak_balance - self._lowest_balance) / self._peak_balance) * 100
        
        # Ortalama odds ve EV
        avg_odds = sum(t.odds for t in self._transactions) / total_bets
        avg_ev = sum(t.expected_value for t in self._transactions) / total_bets
        
        return WalletStats(
            initial_balance=self.initial_balance,
            current_balance=round(self.balance, 2),
            total_bets=total_bets,
            total_wins=self._wins,
            total_losses=self._losses,
            total_staked=round(self._total_staked, 2),
            total_returns=round(self._total_returns, 2),
            total_pnl=round(total_pnl, 2),
            roi=round(roi, 2),
            hit_rate=round(hit_rate, 2),
            peak_balance=round(self._peak_balance, 2),
            lowest_balance=round(self._lowest_balance, 2),
            max_drawdown=round(max_drawdown, 2),
            avg_odds=round(avg_odds, 2),
            avg_ev=round(avg_ev, 3)
        )
    
    def get_transactions(self) -> List[Dict]:
        """Tüm işlemleri döndürür"""
        return [t.to_dict() for t in self._transactions]
    
    def get_balance_history(self) -> List[Dict]:
        """Bakiye geçmişini döndürür"""
        history = [{"index": 0, "balance": self.initial_balance, "date": "start"}]
        
        for i, t in enumerate(self._transactions, 1):
            history.append({
                "index": i,
                "balance": t.balance_after,
                "date": t.date,
                "match": f"{t.home_team} vs {t.away_team}",
                "pnl": t.pnl
            })
        
        return history
    
    def export_to_json(self, path: Path) -> None:
        """
        Tüm verileri JSON dosyasına kaydeder.
        
        Args:
            path: Çıktı dosyası yolu
        """
        data = {
            "summary": self.get_summary().to_dict(),
            "transactions": self.get_transactions(),
            "balance_history": self.get_balance_history(),
            "export_date": datetime.now().isoformat()
        }
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Wallet verileri kaydedildi: {path}")
    
    def print_summary(self) -> None:
        """Özeti konsola yazdırır"""
        stats = self.get_summary()
        
        print("\n" + "=" * 50)
        print("💰 KASA ÖZETİ")
        print("=" * 50)
        print(f"Başlangıç Bakiye: {stats.initial_balance:.2f}")
        print(f"Mevcut Bakiye:    {stats.current_balance:.2f}")
        print(f"Toplam PnL:       {stats.total_pnl:+.2f}")
        print("-" * 50)
        print(f"Toplam Bahis:     {stats.total_bets}")
        print(f"Kazanılan:        {stats.total_wins} ({stats.hit_rate:.1f}%)")
        print(f"Kaybedilen:       {stats.total_losses}")
        print("-" * 50)
        print(f"Toplam Yatırılan: {stats.total_staked:.2f}")
        print(f"Toplam Dönüş:     {stats.total_returns:.2f}")
        print(f"ROI:              {stats.roi:+.2f}%")
        print("-" * 50)
        print(f"En Yüksek Bakiye: {stats.peak_balance:.2f}")
        print(f"En Düşük Bakiye:  {stats.lowest_balance:.2f}")
        print(f"Max Drawdown:     {stats.max_drawdown:.2f}%")
        print("-" * 50)
        print(f"Ortalama Oran:    {stats.avg_odds:.2f}")
        print(f"Ortalama EV:      {stats.avg_ev:.3f}")
        print("=" * 50 + "\n")
    
    def __repr__(self) -> str:
        return f"Wallet(balance={self.balance:.2f}, bets={len(self._transactions)})"
