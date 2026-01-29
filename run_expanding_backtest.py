"""
The Oracle - Expanding Window Backtest
======================================

Bu script her lig için sezon sezon:
1. Önceki sezonlarla eğitir
2. Sonraki sezonu simüle eder
3. Her sezon için ayrı rapor kaydeder
4. Bir lig bitince diğerine geçer

Kullanım:
    cd c:/Users/ahmet/Desktop/Oracle
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
    """
    3 farklı XGBoost modeli eğit:
    - match_result: 1X2 (Kim Kazanır?)
    - over_under: 2.5 Üst/Alt
    - btts: Karşılıklı Gol
    """
    # Dixon-Coles (sadece 1X2 için)
    dixon = DixonColesModel()
    dixon.fit(train_df)
    
    # XGBoost - Match Result (1X2)
    xgb_match = XGBoostPredictor()
    xgb_match.fit(train_df, target_type='match_result')
    
    # XGBoost - Over/Under (2.5 Gol)
    xgb_ou = XGBoostPredictor()
    xgb_ou.fit(train_df, target_type='over_under')
    
    # XGBoost - BTTS (Karşılıklı Gol)
    xgb_btts = XGBoostPredictor()
    xgb_btts.fit(train_df, target_type='btts')
    
    return {
        'dixon': dixon,
        'match_result': xgb_match,
        'over_under': xgb_ou,
        'btts': xgb_btts
    }


def simulate_season(
    test_df: pd.DataFrame,
    models: dict,
    wallet: Wallet,
    value_calc: ValueCalculator
):
    """
    Bir sezonu simüle et - HER MAÇ İÇİN 3 BAHİS
    
    Models: {'dixon': DixonColes, 'match_result': XGB, 'over_under': XGB, 'btts': XGB}
    
    Her maç için:
    1. 1X2 bahsi (match_result modeli)
    2. Over/Under bahsi (over_under modeli)
    3. BTTS bahsi (btts modeli)
    """
    results = []
    STAKE = 10.0  # Sabit bahis miktarı
    
    dixon = models['dixon']
    xgb_match = models['match_result']
    xgb_ou = models['over_under']
    xgb_btts = models['btts']
    
    match_counter = 0
    
    for _, match in test_df.iterrows():
        try:
            match_counter += 1
            home = match['home_team']
            away = match['away_team']
            actual_result = match['result']  # H, D, A
            fthg = int(match.get('fthg', 0))  # Ev sahibi golü
            ftag = int(match.get('ftag', 0))  # Deplasman golü
            total_goals = fthg + ftag
            match_date = str(match.get('date', ''))
            
            # Gerçek sonuçları hesapla
            actual_over = total_goals > 2.5
            actual_btts = (fthg > 0) and (ftag > 0)
            
            match_bets = []
            
            # ============================================
            # 1. BAHİS: MAÇIN SONUCU (1X2)
            # ============================================
            dc_probs = dixon.predict_match_result(home, away)
            xgb_probs = xgb_match.predict_proba(home, away)
            
            # Ensemble (0.4 DC + 0.6 XGB)
            ensemble_1x2 = {
                'home_win': dc_probs['home_win'] * 0.4 + xgb_probs.get('home_win', 0.33) * 0.6,
                'draw': dc_probs['draw'] * 0.4 + xgb_probs.get('draw', 0.33) * 0.6,
                'away_win': dc_probs['away_win'] * 0.4 + xgb_probs.get('away_win', 0.33) * 0.6,
            }
            
            # En yüksek olasılık ile bahis yap
            best_1x2 = max(ensemble_1x2, key=ensemble_1x2.get)
            best_prob_1x2 = ensemble_1x2[best_1x2]
            pred_map = {'home_win': 'H', 'draw': 'D', 'away_win': 'A'}
            bet_type_map = {'home_win': 'MS1', 'draw': 'MS0', 'away_win': 'MS2'}
            predicted_1x2 = pred_map[best_1x2]
            
            # Oran hesapla (margin 1.05)
            odds_1x2 = 1.05 / best_prob_1x2 if best_prob_1x2 > 0 else 2.0
            won_1x2 = (actual_result == predicted_1x2)
            
            # Wallet üzerinden bahis yap
            tx_1x2 = wallet.place_bet(
                match_id=match_counter,
                bet_type=bet_type_map[best_1x2],
                odds=odds_1x2,
                won=won_1x2,
                predicted_prob=best_prob_1x2,
                home_team=home,
                away_team=away,
                date=match_date,
                actual_result=actual_result
            )
            pnl_1x2 = tx_1x2.pnl if tx_1x2 else 0
            
            match_bets.append({
                'type': '1X2',
                'prediction': predicted_1x2,
                'probability': best_prob_1x2,
                'odds': odds_1x2,
                'won': won_1x2,
                'pnl': pnl_1x2
            })
            
            # ============================================
            # 2. BAHİS: OVER/UNDER 2.5
            # ============================================
            ou_probs = xgb_ou.predict_proba(home, away)
            over_prob = ou_probs.get('over', 0.5)
            under_prob = ou_probs.get('under', 0.5)
            
            # En yüksek olasılık ile bahis yap
            if over_prob >= under_prob:
                predicted_ou = 'OVER'
                best_prob_ou = over_prob
                won_ou = actual_over
                bet_type_ou = 'O2.5'
            else:
                predicted_ou = 'UNDER'
                best_prob_ou = under_prob
                won_ou = not actual_over
                bet_type_ou = 'U2.5'
            
            odds_ou = 1.05 / best_prob_ou if best_prob_ou > 0 else 2.0
            
            tx_ou = wallet.place_bet(
                match_id=match_counter,
                bet_type=bet_type_ou,
                odds=odds_ou,
                won=won_ou,
                predicted_prob=best_prob_ou,
                home_team=home,
                away_team=away,
                date=match_date,
                actual_result=f"{fthg}-{ftag}"
            )
            pnl_ou = tx_ou.pnl if tx_ou else 0
            
            match_bets.append({
                'type': 'O/U 2.5',
                'prediction': predicted_ou,
                'probability': best_prob_ou,
                'odds': odds_ou,
                'won': won_ou,
                'pnl': pnl_ou
            })
            
            # ============================================
            # 3. BAHİS: BTTS (KARŞILIKLI GOL)
            # ============================================
            btts_probs = xgb_btts.predict_proba(home, away)
            btts_yes_prob = btts_probs.get('yes', 0.5)
            btts_no_prob = btts_probs.get('no', 0.5)
            
            # En yüksek olasılık ile bahis yap
            if btts_yes_prob >= btts_no_prob:
                predicted_btts = 'VAR'
                best_prob_btts = btts_yes_prob
                won_btts = actual_btts
                bet_type_btts = 'BTTS_Y'
            else:
                predicted_btts = 'YOK'
                best_prob_btts = btts_no_prob
                won_btts = not actual_btts
                bet_type_btts = 'BTTS_N'
            
            odds_btts = 1.05 / best_prob_btts if best_prob_btts > 0 else 2.0
            
            tx_btts = wallet.place_bet(
                match_id=match_counter,
                bet_type=bet_type_btts,
                odds=odds_btts,
                won=won_btts,
                predicted_prob=best_prob_btts,
                home_team=home,
                away_team=away,
                date=match_date,
                actual_result=f"BTTS:{'Y' if actual_btts else 'N'}"
            )
            pnl_btts = tx_btts.pnl if tx_btts else 0
            
            match_bets.append({
                'type': 'BTTS',
                'prediction': predicted_btts,
                'probability': best_prob_btts,
                'odds': odds_btts,
                'won': won_btts,
                'pnl': pnl_btts
            })
            
            # DETAYLI LOGLAMA
            total_pnl = pnl_1x2 + pnl_ou + pnl_btts
            wins_count = sum(1 for b in match_bets if b['won'])
            
            logger.info(f"""
[MAÇ ANALİZİ] {home} vs {away} (Skor: {fthg}-{ftag})
------------------------------------------------
1. 1X2 Tahmini : {predicted_1x2} ({best_prob_1x2:.1%}) @{odds_1x2:.2f} → {'✅' if won_1x2 else '❌'}
2. 2.5 Üst/Alt : {predicted_ou} ({best_prob_ou:.1%}) @{odds_ou:.2f} → {'✅' if won_ou else '❌'}
3. KG Var/Yok  : {predicted_btts} ({best_prob_btts:.1%}) @{odds_btts:.2f} → {'✅' if won_btts else '❌'}
------------------------------------------------
>> ÖZET: {wins_count}/3 doğru | P/L: {total_pnl:+.2f} | Bakiye: {wallet.balance:.2f}
------------------------------------------------""")
            
            results.append({
                'date': match_date,
                'home': home,
                'away': away,
                'score': f"{fthg}-{ftag}",
                'actual_result': actual_result,
                'actual_over': actual_over,
                'actual_btts': actual_btts,
                'bets': match_bets,
                'total_pnl': total_pnl,
                'balance': wallet.balance
            })
            
        except Exception as e:
            logger.error(f"Maç hatası: {home} vs {away}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
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
        
        # Modelleri eğit (3 farklı model)
        try:
            models = train_models(train_df)
            print(f"    ✓ 4 Model eğitildi: Dixon-Coles + 3 XGBoost (1X2, O/U, BTTS)")
        except Exception as e:
            print(f"    ✗ Eğitim hatası: {e}")
            continue
        
        # Simülasyon (her maç için 3 bahis)
        season_start_balance = wallet.balance
        results = simulate_season(test_df, models, wallet, value_calc)
        
        # İstatistikler (yeni yapıya göre)
        total_matches = len(results)
        total_bets = total_matches * 3  # Her maç için 3 bahis
        
        # Tüm bahisleri topla
        all_bets = []
        for r in results:
            all_bets.extend(r.get('bets', []))
        
        total_wins = sum(1 for b in all_bets if b['won'])
        hit_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        
        # 1X2 doğruluk
        correct_1x2 = sum(1 for r in results for b in r.get('bets', []) if b['type'] == '1X2' and b['won'])
        
        season_pnl = wallet.balance - season_start_balance
        total_staked = total_bets * 10
        roi = (season_pnl / total_staked * 100) if total_staked > 0 else 0
        
        # Rapor
        season_report = {
            'league': division,
            'season': test_season,
            'train_seasons': train_seasons,
            'matches_tested': total_matches,
            'total_bets': total_bets,
            'bets_won': total_wins,
            '1x2_correct': correct_1x2,
            '1x2_accuracy': round(correct_1x2 / total_matches * 100, 2) if total_matches > 0 else 0,
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
        if total_matches > 0:
            print(f"    1X2 Doğruluğu: {correct_1x2}/{total_matches} ({correct_1x2/total_matches*100:.1f}%)")
        else:
            print(f"    1X2 Doğruluğu: 0/0 (Veri yok)")
        print(f"    Toplam Bahis: {total_bets}, Kazanan: {total_wins} ({hit_rate:.1f}%)")
        print(f"    PnL: {season_pnl:+.2f} TL, ROI: {roi:+.2f}%")
        print(f"    Bakiye: {wallet.balance:.2f} TL")
        print(f"    📄 Rapor: {report_file.name}")
    
    # Lig özeti
    print(f"\n{'='*60}")
    print(f"📈 {division} LİGİ ÖZET")
    print(f"{'='*60}")
    
    total_bets_all = sum(s.get('total_bets', 0) for s in league_summary)
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
