"""
The Oracle - Backtest Engine

Bu modül, modellerin geçmiş veriler üzerindeki performansını
Walk-Forward Validation yöntemiyle test eder.

Walk-Forward Validation:
    1. 2014-2018 sezonlarıyla eğit → 2019 sezonunu test et
    2. 2014-2019 sezonlarıyla eğit → 2020 sezonunu test et
    3. 2014-2020 sezonlarıyla eğit → 2021 sezonunu test et
    ... ve devam eder

Bu yöntem, gerçek dünya senaryosunu simüle eder:
Model sadece geçmiş veriyle eğitilir, gelecek veriyi "görmez".

Kullanım:
    from backend.simulation.backtest_engine import BacktestEngine
    
    engine = BacktestEngine()
    results = engine.run_backtest()
    engine.print_report()
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
from loguru import logger

from backend.core.config import settings
from backend.database.db_manager import get_database
from backend.models.dixon_coles import DixonColesModel
from backend.models.xgboost_model import XGBoostPredictor
from backend.services.value_calculator import ValueCalculator
from backend.simulation.wallet import Wallet


@dataclass
class BacktestConfig:
    """Backtest konfigürasyonu"""
    initial_balance: float = 1000.0
    stake: float = 10.0
    value_threshold: float = 1.05
    min_edge: float = 0.03
    dixon_weight: float = 0.4
    xgboost_weight: float = 0.6
    bet_on_value_only: bool = True  # Sadece value bet'lere bahis yap
    max_bets_per_day: int = 3  # Günlük maksimum bahis


@dataclass
class SeasonResult:
    """Tek sezon backtest sonucu"""
    season: str
    matches_tested: int
    bets_placed: int
    wins: int
    losses: int
    total_staked: float
    total_pnl: float
    roi: float
    hit_rate: float
    ending_balance: float


@dataclass 
class BacktestReport:
    """Toplam backtest raporu"""
    start_date: str
    end_date: str
    total_seasons: int
    total_matches: int
    total_bets: int
    total_wins: int
    total_losses: int
    total_staked: float
    total_pnl: float
    overall_roi: float
    overall_hit_rate: float
    final_balance: float
    max_drawdown: float
    season_results: List[SeasonResult]
    
    def to_dict(self) -> Dict:
        return {
            **{k: v for k, v in self.__dict__.items() if k != 'season_results'},
            'season_results': [
                {k: v for k, v in sr.__dict__.items()} 
                for sr in self.season_results
            ]
        }



class BacktestEngine:
    """
    Walk-Forward Backtest Motoru
    
    Modellerin geçmiş performansını simüle eder.
    Her sezon için modeli yeniden eğitir ve sonraki sezonu test eder.
    
    Attributes:
        wallet (Wallet): Sanal kasa
        value_calc (ValueCalculator): Value hesaplayıcı
        config (BacktestConfig): Konfigürasyon
        
    Example:
        >>> engine = BacktestEngine()
        >>> report = engine.run_backtest()
        >>> print(f"ROI: {report.overall_roi:.2f}%")
    """
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None
    ) -> None:
        """
        BacktestEngine'i başlatır.
        
        Args:
            config: Backtest konfigürasyonu
        """
        self.config = config or BacktestConfig()
        
        self.wallet = Wallet(
            initial_balance=self.config.initial_balance,
            stake=self.config.stake,
            value_threshold=self.config.value_threshold
        )
        
        self.value_calc = ValueCalculator(
            threshold=self.config.value_threshold,
            min_edge=self.config.min_edge
        )
        
        self._db = None
        self._dixon_coles: Optional[DixonColesModel] = None
        self._xgboost: Optional[XGBoostPredictor] = None
        self._season_results: List[SeasonResult] = []
        
        logger.info("BacktestEngine başlatıldı")
    
    def _get_db(self):
        """Veritabanı bağlantısını döndürür"""
        if self._db is None:
            self._db = get_database()
        return self._db
    
    def _get_seasons(self) -> List[str]:
        """Veritabanındaki sezonları döndürür"""
        db = self._get_db()
        result = db.execute_query("""
            SELECT DISTINCT season FROM matches_history
            WHERE season IS NOT NULL
            ORDER BY season
        """)
        return [r['season'] for r in result]
    
    def _get_season_data(self, season: str) -> pd.DataFrame:
        """Belirli bir sezonun verilerini döndürür"""
        db = self._get_db()
        result = db.execute_query(
            "SELECT * FROM matches_history WHERE season = ? ORDER BY date",
            (season,)
        )
        return pd.DataFrame(result)
    
    def _get_training_data(self, end_season: str) -> pd.DataFrame:
        """Belirli sezonu DAHIL etmeden önceki tüm verileri döndürür"""
        db = self._get_db()
        result = db.execute_query(
            "SELECT * FROM matches_history WHERE season < ? ORDER BY date",
            (end_season,)
        )
        return pd.DataFrame(result)
    
    def _train_models(self, train_df: pd.DataFrame) -> bool:
        """
        Modelleri eğitir.
        
        Args:
            train_df: Eğitim verisi
            
        Returns:
            bool: Başarılı ise True
        """
        try:
            # Dixon-Coles
            self._dixon_coles = DixonColesModel()
            self._dixon_coles.fit(train_df)
            
            # XGBoost
            self._xgboost = XGBoostPredictor()
            self._xgboost.fit(train_df)
            
            return True
            
        except Exception as e:
            logger.error(f"Model eğitim hatası: {e}")
            return False
    
    def _predict_match(
        self, 
        home_team: str, 
        away_team: str
    ) -> Dict[str, float]:
        """
        Ensemble tahmin yapar.
        
        Returns:
            Dict: {'home_win': float, 'draw': float, 'away_win': float}
        """
        # Dixon-Coles tahmini
        dc_result = self._dixon_coles.predict_match_result(home_team, away_team)
        
        # XGBoost tahmini
        try:
            xgb_result = self._xgboost.predict_proba(home_team, away_team)
        except Exception:
            # Bilinmeyen takım durumunda sadece Dixon-Coles kullan
            return dc_result
        
        # Ensemble
        weights = (self.config.dixon_weight, self.config.xgboost_weight)
        total_weight = sum(weights)
        
        ensemble = {
            'home_win': (
                dc_result['home_win'] * weights[0] +
                xgb_result['home_win'] * weights[1]
            ) / total_weight,
            'draw': (
                dc_result['draw'] * weights[0] +
                xgb_result['draw'] * weights[1]
            ) / total_weight,
            'away_win': (
                dc_result['away_win'] * weights[0] +
                xgb_result['away_win'] * weights[1]
            ) / total_weight
        }
        
        return ensemble
    
    def _simulate_match(
        self,
        match: Dict,
        predictions: Dict[str, float]
    ) -> Optional[Dict]:
        """
        Tek bir maç için bahis simülasyonu yapar.
        
        Args:
            match: Maç verisi (veritabanından)
            predictions: Model tahminleri
            
        Returns:
            Dict: Bahis sonucu veya None
        """
        home_team = match['home_team']
        away_team = match['away_team']
        actual_result = match['result']  # H, D, A
        date = match['date']
        match_id = match['id']
        
        # Bahis oranları (API'den gelen veri yoksa simüle et)
        home_odds = match.get('home_odds')
        draw_odds = match.get('draw_odds')
        away_odds = match.get('away_odds')
        
        # Oranlar yoksa, olasılıklardan hesapla (margin ekle)
        margin = 1.05  # %5 bahisçi marjı
        if home_odds is None or draw_odds is None or away_odds is None:
            home_odds = margin / predictions['home_win'] if predictions['home_win'] > 0 else 10
            draw_odds = margin / predictions['draw'] if predictions['draw'] > 0 else 5
            away_odds = margin / predictions['away_win'] if predictions['away_win'] > 0 else 4
        
        # En yüksek value'yu bul
        best_bet = None
        best_ev = 0
        
        # Ev sahibi analizi
        ev_home = self.value_calc.calculate_ev(predictions['home_win'], home_odds)
        if self.value_calc.is_value_bet(predictions['home_win'], home_odds):
            if ev_home > best_ev:
                best_bet = ('MS1', predictions['home_win'], home_odds, 'H')
                best_ev = ev_home
        
        # Beraberlik analizi
        ev_draw = self.value_calc.calculate_ev(predictions['draw'], draw_odds)
        if self.value_calc.is_value_bet(predictions['draw'], draw_odds):
            if ev_draw > best_ev:
                best_bet = ('MS0', predictions['draw'], draw_odds, 'D')
                best_ev = ev_draw
        
        # Deplasman analizi
        ev_away = self.value_calc.calculate_ev(predictions['away_win'], away_odds)
        if self.value_calc.is_value_bet(predictions['away_win'], away_odds):
            if ev_away > best_ev:
                best_bet = ('MS2', predictions['away_win'], away_odds, 'A')
                best_ev = ev_away
        
        # Value bet yoksa bahis yapma
        if best_bet is None and self.config.bet_on_value_only:
            return None
        
        # Bakiye kontrolü
        if not self.wallet.can_bet():
            return None
        
        # Bahis yap
        if best_bet:
            bet_type, prob, odds, win_condition = best_bet
            won = (actual_result == win_condition)
            
            transaction = self.wallet.place_bet(
                match_id=match_id,
                bet_type=bet_type,
                odds=odds,
                won=won,
                predicted_prob=prob,
                home_team=home_team,
                away_team=away_team,
                date=date,
                actual_result=actual_result
            )
            
            return {
                'match_id': match_id,
                'home_team': home_team,
                'away_team': away_team,
                'bet_type': bet_type,
                'odds': odds,
                'probability': prob,
                'ev': best_ev,
                'won': won,
                'pnl': transaction.pnl if transaction else 0
            }
        
        return None
    
    def run_backtest(
        self,
        min_train_seasons: int = 2
    ) -> BacktestReport:
        """
        Walk-Forward Backtest çalıştırır.
        
        Args:
            min_train_seasons: Minimum eğitim sezonu sayısı
            
        Returns:
            BacktestReport: Backtest sonuçları
        """
        logger.info("=" * 50)
        logger.info("WALK-FORWARD BACKTEST BAŞLIYOR")
        logger.info("=" * 50)
        
        seasons = self._get_seasons()
        
        if len(seasons) < min_train_seasons + 1:
            raise ValueError(
                f"En az {min_train_seasons + 1} sezon gerekli! "
                f"Mevcut: {len(seasons)}"
            )
        
        logger.info(f"Toplam {len(seasons)} sezon bulundu: {seasons}")
        
        # Walk-Forward döngüsü
        self._season_results = []
        
        for i in range(min_train_seasons, len(seasons)):
            test_season = seasons[i]
            train_seasons = seasons[:i]
            
            logger.info(f"\n--- Test Sezonu: {test_season} ---")
            logger.info(f"Eğitim: {train_seasons}")
            
            # Eğitim verisi
            train_df = self._get_training_data(test_season)
            
            if train_df.empty:
                logger.warning(f"Eğitim verisi boş, {test_season} atlanıyor")
                continue
            
            logger.info(f"Eğitim verisi: {len(train_df)} maç")
            
            # Modelleri eğit
            if not self._train_models(train_df):
                logger.warning(f"Model eğitilemedi, {test_season} atlanıyor")
                continue
            
            # Test verisi
            test_df = self._get_season_data(test_season)
            
            if test_df.empty:
                logger.warning(f"Test verisi boş, {test_season} atlanıyor")
                continue
            
            logger.info(f"Test verisi: {len(test_df)} maç")
            
            # Sezon başlangıç bakiyesi
            season_start_balance = self.wallet.balance
            season_bets = 0
            season_wins = 0
            season_losses = 0
            
            # Her maçı test et
            for _, match in test_df.iterrows():
                try:
                    predictions = self._predict_match(
                        match['home_team'],
                        match['away_team']
                    )
                    
                    result = self._simulate_match(match.to_dict(), predictions)
                    
                    if result:
                        season_bets += 1
                        if result['won']:
                            season_wins += 1
                        else:
                            season_losses += 1
                            
                except Exception as e:
                    logger.debug(f"Maç hatası: {e}")
                    continue
            
            # Sezon özeti
            season_pnl = self.wallet.balance - season_start_balance
            season_staked = season_bets * self.config.stake
            season_roi = (season_pnl / season_staked * 100) if season_staked > 0 else 0
            season_hit_rate = (season_wins / season_bets * 100) if season_bets > 0 else 0
            
            season_result = SeasonResult(
                season=test_season,
                matches_tested=len(test_df),
                bets_placed=season_bets,
                wins=season_wins,
                losses=season_losses,
                total_staked=season_staked,
                total_pnl=round(season_pnl, 2),
                roi=round(season_roi, 2),
                hit_rate=round(season_hit_rate, 2),
                ending_balance=round(self.wallet.balance, 2)
            )
            
            self._season_results.append(season_result)
            
            logger.info(
                f"Sezon {test_season}: "
                f"{season_bets} bahis, "
                f"{season_wins} kazanç, "
                f"ROI: {season_roi:+.2f}%, "
                f"Bakiye: {self.wallet.balance:.2f}"
            )
        
        # Toplam rapor
        stats = self.wallet.get_summary()
        
        report = BacktestReport(
            start_date=seasons[0],
            end_date=seasons[-1],
            total_seasons=len(self._season_results),
            total_matches=sum(sr.matches_tested for sr in self._season_results),
            total_bets=stats.total_bets,
            total_wins=stats.total_wins,
            total_losses=stats.total_losses,
            total_staked=stats.total_staked,
            total_pnl=stats.total_pnl,
            overall_roi=stats.roi,
            overall_hit_rate=stats.hit_rate,
            final_balance=stats.current_balance,
            max_drawdown=stats.max_drawdown,
            season_results=self._season_results
        )
        
        logger.info("\n" + "=" * 50)
        logger.info("BACKTEST TAMAMLANDI")
        logger.info("=" * 50)
        
        return report
    
    def print_report(self, report: Optional[BacktestReport] = None) -> None:
        """Raporu konsola yazdırır"""
        if report is None:
            if not self._season_results:
                print("Henüz backtest çalıştırılmadı!")
                return
            stats = self.wallet.get_summary()
            report = BacktestReport(
                start_date="",
                end_date="",
                total_seasons=len(self._season_results),
                total_matches=sum(sr.matches_tested for sr in self._season_results),
                total_bets=stats.total_bets,
                total_wins=stats.total_wins,
                total_losses=stats.total_losses,
                total_staked=stats.total_staked,
                total_pnl=stats.total_pnl,
                overall_roi=stats.roi,
                overall_hit_rate=stats.hit_rate,
                final_balance=stats.current_balance,
                max_drawdown=stats.max_drawdown,
                season_results=self._season_results
            )
        
        print("\n" + "=" * 60)
        print("📊 BACKTEST RAPORU")
        print("=" * 60)
        
        print(f"\n📅 Dönem: {report.start_date} → {report.end_date}")
        print(f"📈 Test Edilen Sezon: {report.total_seasons}")
        print(f"⚽ Toplam Maç: {report.total_matches}")
        
        print("\n" + "-" * 60)
        print("💰 FİNANSAL ÖZET")
        print("-" * 60)
        print(f"Başlangıç Bakiye:  {self.config.initial_balance:.2f}")
        print(f"Final Bakiye:      {report.final_balance:.2f}")
        print(f"Toplam PnL:        {report.total_pnl:+.2f}")
        print(f"Toplam Yatırılan:  {report.total_staked:.2f}")
        
        print("\n" + "-" * 60)
        print("📈 PERFORMANS METRİKLERİ")
        print("-" * 60)
        print(f"Toplam Bahis:      {report.total_bets}")
        print(f"Kazanılan:         {report.total_wins} ({report.overall_hit_rate:.1f}%)")
        print(f"Kaybedilen:        {report.total_losses}")
        print(f"ROI:               {report.overall_roi:+.2f}%")
        print(f"Max Drawdown:      {report.max_drawdown:.2f}%")
        
        # Hedef kontrolü
        print("\n" + "-" * 60)
        print("🎯 HEDEF KONTROLÜ")
        print("-" * 60)
        
        roi_ok = "✅" if report.overall_roi > 5 else "❌"
        hit_ok = "✅" if report.overall_hit_rate > 55 else "❌"
        dd_ok = "✅" if report.max_drawdown < 20 else "❌"
        
        print(f"{roi_ok} ROI > %5:          {report.overall_roi:.2f}%")
        print(f"{hit_ok} Hit Rate > %55:    {report.overall_hit_rate:.2f}%")
        print(f"{dd_ok} Max DD < %20:       {report.max_drawdown:.2f}%")
        
        # Sezon detayları
        print("\n" + "-" * 60)
        print("📋 SEZON DETAYLARI")
        print("-" * 60)
        print(f"{'Sezon':<12} {'Bahis':>6} {'Kazanç':>8} {'ROI':>8} {'Bakiye':>10}")
        print("-" * 60)
        
        for sr in report.season_results:
            print(
                f"{sr.season:<12} "
                f"{sr.bets_placed:>6} "
                f"{sr.wins:>8} "
                f"{sr.roi:>+7.2f}% "
                f"{sr.ending_balance:>10.2f}"
            )
        
        print("=" * 60 + "\n")
    
    def export_report(self, path: Optional[Path] = None) -> Path:
        """
        Raporu JSON dosyasına kaydeder.
        
        Args:
            path: Çıktı dosyası yolu
            
        Returns:
            Path: Kaydedilen dosya yolu
        """
        if path is None:
            path = settings.LOGS_DIR / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.wallet.get_summary()
        
        report_data = {
            'config': {
                'initial_balance': self.config.initial_balance,
                'stake': self.config.stake,
                'value_threshold': self.config.value_threshold,
                'min_edge': self.config.min_edge,
                'dixon_weight': self.config.dixon_weight,
                'xgboost_weight': self.config.xgboost_weight
            },
            'summary': stats.to_dict(),
            'season_results': [
                {k: v for k, v in sr.__dict__.items()}
                for sr in self._season_results
            ],
            'transactions': self.wallet.get_transactions(),
            'balance_history': self.wallet.get_balance_history(),
            'generated_at': datetime.now().isoformat()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Backtest raporu kaydedildi: {path}")
        
        return path
    
    def close(self) -> None:
        """Kaynakları temizler"""
        if self._db:
            self._db.close()
            self._db = None


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="The Oracle - Backtest Engine")
    parser.add_argument('--balance', type=float, default=1000, help='Başlangıç bakiyesi')
    parser.add_argument('--stake', type=float, default=10, help='Bahis miktarı')
    parser.add_argument('--threshold', type=float, default=1.05, help='EV eşiği')
    parser.add_argument('--export', action='store_true', help='Raporu kaydet')
    
    args = parser.parse_args()
    
    config = BacktestConfig(
        initial_balance=args.balance,
        stake=args.stake,
        value_threshold=args.threshold
    )
    
    engine = BacktestEngine(config)
    
    try:
        report = engine.run_backtest()
        engine.print_report(report)
        
        if args.export:
            path = engine.export_report()
            print(f"📁 Rapor kaydedildi: {path}")
            
    finally:
        engine.close()


if __name__ == "__main__":
    main()
