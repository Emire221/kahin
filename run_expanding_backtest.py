"""
The Oracle - Expanding Window Backtest
======================================

Bu script her lig için sezon sezon:
1. Önceki sezonlarla eğitir
2. Sonraki sezonu simüle eder
3. Her sezon için ayrı rapor kaydeder
4. Bir lig bitince diğerine geçer

Kullanım:
    cd c:\Users\ahmet\Desktop\Oracle
    python run_expanding_backtest.py
    
veya belirli bir lig için:
    python run_expanding_backtest.py --league T1
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from io import StringIO

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from loguru import logger

from backend.database.db_manager import get_database
from backend.data.data_loader import DataLoader
from backend.models.dixon_coles import DixonColesModel
from backend.models.xgboost_model import XGBoostPredictor
from backend.services.value_calculator import ValueCalculator
from backend.simulation.wallet import Wallet


# Ligler ve sezonlar
TIER_1_LEAGUES = ['T1', 'E0', 'D1', 'I1', 'SP1', 'F1', 'N1', 'B1', 'P1']
SEASONS = [
    '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020',
    '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025'
]


def setup_logging(log_file: Path):
    """Log dosyası ayarla"""
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(log_file, level="DEBUG", rotation="10 MB")
    return log_file


def load_league_data(division: str) -> pd.DataFrame:
    """Belirli bir lig için tüm verileri yükle"""
    db = get_database()
    result = db.execute_query(
        "SELECT * FROM matches_history WHERE division = ? ORDER BY date",
        (division,)
    )
    return pd.DataFrame(result)


def get_season_data(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Belirli sezonu filtrele"""
    return df[df['season'] == season].copy()


def get_training_data(df: pd.DataFrame, seasons: list) -> pd.DataFrame:
    """Belirtilen sezonları birleştir"""
    return df[df['season'].isin(seasons)].copy()


def train_models(train_df: pd.DataFrame):
    """Modelleri eğit"""
    # Dixon-Coles
    dixon = DixonColesModel()
    dixon.fit(train_df)
    
    # XGBoost
    xgb = XGBoostPredictor()
    xgb.fit(train_df)
    
    return dixon, xgb


def simulate_season(
    test_df: pd.DataFrame,
    dixon: DixonColesModel,
    xgb: XGBoostPredictor,
    wallet: Wallet,
    value_calc: ValueCalculator
):
    """Bir sezonu simüle et"""
    results = []
    
    for _, match in test_df.iterrows():
        try:
            home = match['home_team']
            away = match['away_team']
            actual = match['result']
            
            # Tahmin yap
            dc_probs = dixon.predict_match_result(home, away)
            xgb_probs = xgb.predict_proba(home, away)
            
            # Ensemble (0.4 DC + 0.6 XGB)
            ensemble = {
                'home_win': dc_probs['home_win'] * 0.4 + xgb_probs['home_win'] * 0.6,
                'draw': dc_probs['draw'] * 0.4 + xgb_probs['draw'] * 0.6,
                'away_win': dc_probs['away_win'] * 0.4 + xgb_probs['away_win'] * 0.6,
            }
            
            # En yüksek olasılık
            pred = max(ensemble, key=ensemble.get)
            pred_map = {'home_win': 'H', 'draw': 'D', 'away_win': 'A'}
            predicted = pred_map[pred]
            
            # Value bahis kontrolü
            odds_map = {
                'H': match.get('home_odds'),
                'D': match.get('draw_odds'),
                'A': match.get('away_odds')
            }
            
            bet_result = None
            for outcome, prob_key in [('H', 'home_win'), ('D', 'draw'), ('A', 'away_win')]:
                odds = odds_map.get(outcome)
                if odds and odds > 0:
                    value = value_calc.calculate_value(ensemble[prob_key], odds)
                    if value and value['has_value']:
                        # Bahis yap
                        won = (actual == outcome)
                        pnl = wallet.place_bet(won, odds)
                        bet_result = {
                            'date': str(match.get('date', '')),
                            'match': f"{home} vs {away}",
                            'bet_on': outcome,
                            'odds': odds,
                            'probability': ensemble[prob_key],
                            'value': value['value'],
                            'won': won,
                            'pnl': pnl,
                            'balance': wallet.balance
                        }
                        break
            
            results.append({
                'date': str(match.get('date', '')),
                'home': home,
                'away': away,
                'predicted': predicted,
                'actual': actual,
                'correct': predicted == actual,
                'bet': bet_result
            })
            
        except Exception as e:
            logger.debug(f"Maç hatası: {e}")
            continue
    
    return results


def run_league_backtest(division: str, output_dir: Path):
    """Tek bir lig için expanding window backtest"""
    
    print(f"\n{'='*60}")
    print(f"🏆 LİG: {division}")
    print(f"{'='*60}")
    
    # Lig dizini oluştur
    league_dir = output_dir / division
    league_dir.mkdir(parents=True, exist_ok=True)
    
    # Lig verilerini yükle
    df = load_league_data(division)
    
    if df.empty:
        print(f"❌ {division} ligi için veri bulunamadı!")
        return
    
    # Mevcut sezonları bul
    available_seasons = sorted(df['season'].dropna().unique().tolist())
    print(f"📅 Mevcut sezonlar: {available_seasons}")
    
    if len(available_seasons) < 2:
        print(f"❌ En az 2 sezon gerekli!")
        return
    
    # Value calculator ve wallet
    value_calc = ValueCalculator(threshold=1.05, min_edge=0.03)
    wallet = Wallet(initial_balance=1000.0, stake=10.0)
    
    league_summary = []
    
    # Expanding window döngüsü
    for i in range(1, len(available_seasons)):
        train_seasons = available_seasons[:i]
        test_season = available_seasons[i]
        
        print(f"\n--- Sezon {i}/{len(available_seasons)-1}: {test_season} ---")
        print(f"    Eğitim: {train_seasons}")
        
        # Log dosyası
        log_file = league_dir / f"{test_season}_log.txt"
        setup_logging(log_file)
        
        # Eğitim verisi
        train_df = get_training_data(df, train_seasons)
        test_df = get_season_data(df, test_season)
        
        if train_df.empty or test_df.empty:
            print(f"    ⚠️ Veri eksik, atlanıyor")
            continue
        
        print(f"    Eğitim: {len(train_df)} maç")
        print(f"    Test: {len(test_df)} maç")
        
        # Modelleri eğit
        try:
            dixon, xgb = train_models(train_df)
            print(f"    ✓ Modeller eğitildi")
        except Exception as e:
            print(f"    ✗ Eğitim hatası: {e}")
            continue
        
        # Simülasyon
        season_start_balance = wallet.balance
        results = simulate_season(test_df, dixon, xgb, wallet, value_calc)
        
        # İstatistikler
        total_predictions = len(results)
        correct = sum(1 for r in results if r['correct'])
        accuracy = (correct / total_predictions * 100) if total_predictions > 0 else 0
        
        bets = [r['bet'] for r in results if r['bet']]
        total_bets = len(bets)
        wins = sum(1 for b in bets if b['won'])
        hit_rate = (wins / total_bets * 100) if total_bets > 0 else 0
        
        season_pnl = wallet.balance - season_start_balance
        total_staked = total_bets * 10
        roi = (season_pnl / total_staked * 100) if total_staked > 0 else 0
        
        # Rapor
        season_report = {
            'league': division,
            'season': test_season,
            'train_seasons': train_seasons,
            'matches_tested': total_predictions,
            'predictions_correct': correct,
            'accuracy': round(accuracy, 2),
            'bets_placed': total_bets,
            'bets_won': wins,
            'hit_rate': round(hit_rate, 2),
            'total_staked': total_staked,
            'pnl': round(season_pnl, 2),
            'roi': round(roi, 2),
            'starting_balance': round(season_start_balance, 2),
            'ending_balance': round(wallet.balance, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        league_summary.append(season_report)
        
        # Raporu kaydet
        report_file = league_dir / f"{test_season}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(season_report, f, indent=2, ensure_ascii=False)
        
        # Ekrana yazdır
        print(f"\n    📊 SEZON SONUCU: {test_season}")
        print(f"    ─────────────────────────────")
        print(f"    Tahmin Doğruluğu: {correct}/{total_predictions} ({accuracy:.1f}%)")
        print(f"    Bahis: {total_bets}, Kazanan: {wins} ({hit_rate:.1f}%)")
        print(f"    PnL: {season_pnl:+.2f} TL, ROI: {roi:+.2f}%")
        print(f"    Bakiye: {wallet.balance:.2f} TL")
        print(f"    📄 Rapor: {report_file.name}")
    
    # Lig özeti
    print(f"\n{'='*60}")
    print(f"📈 {division} LİGİ ÖZET")
    print(f"{'='*60}")
    
    total_bets_all = sum(s['bets_placed'] for s in league_summary)
    total_wins_all = sum(s['bets_won'] for s in league_summary)
    total_pnl_all = sum(s['pnl'] for s in league_summary)
    total_staked_all = sum(s['total_staked'] for s in league_summary)
    overall_roi = (total_pnl_all / total_staked_all * 100) if total_staked_all > 0 else 0
    
    print(f"Toplam Sezon: {len(league_summary)}")
    print(f"Toplam Bahis: {total_bets_all}")
    print(f"Toplam Kazanan: {total_wins_all}")
    print(f"Toplam PnL: {total_pnl_all:+.2f} TL")
    print(f"Toplam ROI: {overall_roi:+.2f}%")
    print(f"Final Bakiye: {wallet.balance:.2f} TL")
    
    # Lig özetini kaydet
    summary_file = league_dir / "league_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'league': division,
            'seasons': league_summary,
            'totals': {
                'total_bets': total_bets_all,
                'total_wins': total_wins_all,
                'total_pnl': round(total_pnl_all, 2),
                'overall_roi': round(overall_roi, 2),
                'final_balance': round(wallet.balance, 2)
            }
        }, f, indent=2, ensure_ascii=False)
    
    return wallet.balance


def main():
    parser = argparse.ArgumentParser(description='Expanding Window Backtest')
    parser.add_argument('--league', '-l', type=str, help='Tek lig için (örn: T1)')
    parser.add_argument('--all', '-a', action='store_true', help='Tüm ligler')
    args = parser.parse_args()
    
    print("=" * 60)
    print("THE ORACLE - EXPANDING WINDOW BACKTEST")
    print("=" * 60)
    print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Çıktı dizini
    output_dir = Path("reports/backtest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Hangi ligler
    if args.league:
        leagues = [args.league.upper()]
    else:
        leagues = TIER_1_LEAGUES
    
    print(f"\n🏆 Ligler: {leagues}")
    
    # Her lig için backtest
    for league in leagues:
        run_league_backtest(league, output_dir)
    
    print("\n" + "=" * 60)
    print("✅ TÜM BACKTEST İŞLEMLERİ TAMAMLANDI!")
    print(f"📁 Raporlar: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
