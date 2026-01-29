"""
The Oracle - Tüm Ligler İçin Tam Backtest (İzole Modeller)
==========================================================

Her lig için 4 AYRÎ model eğitilir:
- Ligler birbirinden TAMAMEN bağımsızdır
- Her lig kendi tarihsel verileriyle eğitilir
- Modeller lig bazında kaydedilir

Kullanım:
    python run_full_backtest.py

Raporlar: reports/backtest/{LIG_KODU}/
Modeller: models/{LIG_KODU}/
"""

import sys
import json
import time
import pickle
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from loguru import logger

from backend.database.db_manager import get_database
from backend.models.dixon_coles import DixonColesModel
from backend.models.xgboost_model import XGBoostPredictor
from backend.simulation.wallet import Wallet


# =============================================
# AYARLAR
# =============================================
ALL_LEAGUES = ['T1', 'E0', 'D1', 'I1', 'SP1', 'F1', 'N1', 'B1', 'P1']
INITIAL_BALANCE = 1000.0
STAKE = 10.0


def setup_logging(log_file: Path):
    """Log dosyası ayarla"""
    logger.remove()
    logger.add(sys.stdout, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")
    logger.add(log_file, level="DEBUG", rotation="50 MB")


def load_league_data(division: str) -> pd.DataFrame:
    """
    SADECE belirtilen ligin verilerini yükle.
    Diğer lig verileri ASLA dahil edilmez.
    """
    db = get_database()
    result = db.execute_query(
        "SELECT * FROM matches_history WHERE division = ? ORDER BY date",
        (division,)
    )
    df = pd.DataFrame(result)
    logger.info(f"[{division}] {len(df)} maç yüklendi (sadece bu lig)")
    return df


def train_league_models(train_df: pd.DataFrame, league: str, model_dir: Path):
    """
    Bir lig için 4 UZMAN model eğit ve kaydet.
    
    Bu modeller SADECE bu ligin verileriyle eğitilir.
    Diğer liglerin verileri kesinlikle kullanılmaz.
    """
    logger.info(f"[{league}] 4 uzman model eğitiliyor ({len(train_df)} maç)")
    
    # 1. Dixon-Coles (1X2 için)
    dixon = DixonColesModel()
    dixon.fit(train_df)
    
    # 2. XGBoost - Match Result (1X2)
    xgb_1x2 = XGBoostPredictor()
    xgb_1x2.fit(train_df, target_type='match_result')
    
    # 3. XGBoost - Over/Under (2.5 Gol)
    xgb_ou = XGBoostPredictor()
    xgb_ou.fit(train_df, target_type='over_under')
    
    # 4. XGBoost - BTTS (Karşılıklı Gol)
    xgb_btts = XGBoostPredictor()
    xgb_btts.fit(train_df, target_type='btts')
    
    models = {
        'dixon': dixon,
        'xgb_1x2': xgb_1x2,
        'xgb_ou': xgb_ou,
        'xgb_btts': xgb_btts
    }
    
    # Modelleri kaydet
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{league}_models.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(models, f)
    logger.info(f"[{league}] Modeller kaydedildi: {model_file}")
    
    return models


def simulate_season(test_df: pd.DataFrame, models: dict, wallet: Wallet, league: str):
    """Her maç için 3 bahis simüle et"""
    results = []
    
    dixon = models['dixon']
    xgb_1x2 = models['xgb_1x2']
    xgb_ou = models['xgb_ou']
    xgb_btts = models['xgb_btts']
    
    match_counter = 0
    
    for _, match in test_df.iterrows():
        try:
            match_counter += 1
            home = match['home_team']
            away = match['away_team']
            actual_result = match['result']
            fthg = int(match.get('fthg', 0))
            ftag = int(match.get('ftag', 0))
            total_goals = fthg + ftag
            match_date = str(match.get('date', ''))
            
            actual_over = total_goals > 2.5
            actual_btts = (fthg > 0) and (ftag > 0)
            
            match_bets = []
            
            # ===== 1. BAHİS: 1X2 =====
            dc_probs = dixon.predict_match_result(home, away)
            xgb_probs = xgb_1x2.predict_proba(home, away)
            
            # Ensemble (0.4 DC + 0.6 XGB)
            ensemble = {
                'home_win': dc_probs['home_win'] * 0.4 + xgb_probs.get('home_win', 0.33) * 0.6,
                'draw': dc_probs['draw'] * 0.4 + xgb_probs.get('draw', 0.33) * 0.6,
                'away_win': dc_probs['away_win'] * 0.4 + xgb_probs.get('away_win', 0.33) * 0.6,
            }
            
            best = max(ensemble, key=ensemble.get)
            prob = ensemble[best]
            pred_map = {'home_win': 'H', 'draw': 'D', 'away_win': 'A'}
            bet_type_map = {'home_win': 'MS1', 'draw': 'MS0', 'away_win': 'MS2'}
            predicted = pred_map[best]
            
            odds = 1.05 / prob if prob > 0 else 2.0
            won = (actual_result == predicted)
            
            tx = wallet.place_bet(
                match_id=match_counter, bet_type=bet_type_map[best],
                odds=odds, won=won, predicted_prob=prob,
                home_team=home, away_team=away, date=match_date, actual_result=actual_result
            )
            pnl = tx.pnl if tx else 0
            match_bets.append({'type': '1X2', 'choice': bet_type_map[best], 'prob': prob, 'won': won, 'pnl': pnl})
            
            # ===== 2. BAHİS: OVER/UNDER =====
            ou_probs = xgb_ou.predict_proba(home, away)
            over = ou_probs.get('over', 0.5)
            under = ou_probs.get('under', 0.5)
            
            if over >= under:
                prob_ou, won_ou, bet_type_ou = over, actual_over, 'O2.5'
            else:
                prob_ou, won_ou, bet_type_ou = under, not actual_over, 'U2.5'
            
            odds_ou = 1.05 / prob_ou if prob_ou > 0 else 2.0
            tx_ou = wallet.place_bet(
                match_id=match_counter, bet_type=bet_type_ou,
                odds=odds_ou, won=won_ou, predicted_prob=prob_ou,
                home_team=home, away_team=away, date=match_date, actual_result=f"{fthg}-{ftag}"
            )
            pnl_ou = tx_ou.pnl if tx_ou else 0
            match_bets.append({'type': 'O/U', 'choice': bet_type_ou, 'prob': prob_ou, 'won': won_ou, 'pnl': pnl_ou})
            
            # ===== 3. BAHİS: BTTS =====
            btts_probs = xgb_btts.predict_proba(home, away)
            yes = btts_probs.get('yes', 0.5)
            no = btts_probs.get('no', 0.5)
            
            if yes >= no:
                prob_btts, won_btts, bet_type_btts = yes, actual_btts, 'BTTS_Y'
            else:
                prob_btts, won_btts, bet_type_btts = no, not actual_btts, 'BTTS_N'
            
            odds_btts = 1.05 / prob_btts if prob_btts > 0 else 2.0
            tx_btts = wallet.place_bet(
                match_id=match_counter, bet_type=bet_type_btts,
                odds=odds_btts, won=won_btts, predicted_prob=prob_btts,
                home_team=home, away_team=away, date=match_date, actual_result=f"BTTS:{'Y' if actual_btts else 'N'}"
            )
            pnl_btts = tx_btts.pnl if tx_btts else 0
            match_bets.append({'type': 'BTTS', 'choice': bet_type_btts, 'prob': prob_btts, 'won': won_btts, 'pnl': pnl_btts})
            
            total_pnl = pnl + pnl_ou + pnl_btts
            wins = sum(1 for b in match_bets if b['won'])
            
            # Detaylı loglama (Kullanıcı isteği üzerine)
            logger.debug(f"[{league}] {home} vs {away}")
            
            # 1X2 Detayı
            b1 = match_bets[0]
            logger.debug(f"   > 1X2 : {b1['choice']:<4} ({b1['prob']:.2%}) | Sonuç: {predicted:<3} | Durum: {'✅' if b1['won'] else '❌'} | PnL: {b1['pnl']:+.2f}")
            
            # O/U Detayı
            b2 = match_bets[1]
            logger.debug(f"   > O/U : {b2['choice']:<4} ({b2['prob']:.2%}) | Sonuç: {'Ov' if actual_over else 'Un':<3} | Durum: {'✅' if b2['won'] else '❌'} | PnL: {b2['pnl']:+.2f}")
            
            # BTTS Detayı
            b3 = match_bets[2]
            logger.debug(f"   > BTTS: {b3['choice']:<4} ({b3['prob']:.2%}) | Sonuç: {'Y' if actual_btts else 'N':<3}  | Durum: {'✅' if b3['won'] else '❌'} | PnL: {b3['pnl']:+.2f}")
            
            logger.debug(f"   = Toplam PnL: {total_pnl:+.2f}")
            
            results.append({'bets': match_bets, 'total_pnl': total_pnl})
            
        except Exception as e:
            logger.debug(f"Maç hatası: {e}")
            continue
    
    return results


def run_league_backtest(league: str, output_dir: Path, model_dir: Path):
    """
    TEK bir lig için TAMAMEN izole backtest.
    
    Bu fonksiyon:
    1. SADECE bu ligin verilerini yükler
    2. SADECE bu lig için 4 model eğitir
    3. SADECE bu ligin maçlarını simüle eder
    4. Sonuçları ve modelleri bu lige özel dizine kaydeder
    """
    
    print(f"\n{'='*60}")
    print(f"🏆 [{league}] LİG BACKTEST BAŞLIYOR")
    print(f"{'='*60}")
    
    # Lig özel dizinler
    league_report_dir = output_dir / league
    league_model_dir = model_dir / league
    league_report_dir.mkdir(parents=True, exist_ok=True)
    league_model_dir.mkdir(parents=True, exist_ok=True)
    
    # Log dosyası
    log_file = league_report_dir / f"{league}_backtest.log"
    setup_logging(log_file)
    
    # SADECE bu ligin verileri
    df = load_league_data(league)
    
    if df.empty:
        print(f"❌ [{league}] Veri yok!")
        return None
    
    seasons = sorted(df['season'].dropna().unique().tolist())
    print(f"📅 [{league}] Sezonlar: {seasons}")
    
    if len(seasons) < 2:
        print(f"❌ [{league}] En az 2 sezon gerekli!")
        return None
    
    wallet = Wallet(initial_balance=INITIAL_BALANCE, stake=STAKE)
    league_summary = []
    
    # Expanding window
    for i in range(1, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]
        
        print(f"\n[{league}] Sezon {i}/{len(seasons)-1}: {test_season}")
        print(f"    Eğitim sezonları: {train_seasons}")
        
        train_df = df[df['season'].isin(train_seasons)].copy()
        test_df = df[df['season'] == test_season].copy()
        
        if train_df.empty or test_df.empty:
            continue
        
        print(f"    Eğitim: {len(train_df)} maç | Test: {len(test_df)} maç")
        
        try:
            # Bu lig için 4 model eğit
            models = train_league_models(train_df, league, league_model_dir)
            print(f"    ✓ [{league}] 4 uzman model eğitildi")
        except Exception as e:
            print(f"    ✗ [{league}] Eğitim hatası: {e}")
            continue
        
        # Simülasyon
        season_start = wallet.balance
        results = simulate_season(test_df, models, wallet, league)
        
        # İstatistikler
        total_matches = len(results)
        total_bets = total_matches * 3
        all_bets = [b for r in results for b in r.get('bets', [])]
        total_wins = sum(1 for b in all_bets if b['won'])
        hit_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        
        season_pnl = wallet.balance - season_start
        roi = (season_pnl / (total_bets * STAKE) * 100) if total_bets > 0 else 0
        
        report = {
            'league': league,
            'season': test_season,
            'train_seasons': train_seasons,
            'matches': total_matches,
            'bets': total_bets,
            'wins': total_wins,
            'hit_rate': round(hit_rate, 2),
            'pnl': round(season_pnl, 2),
            'roi': round(roi, 2),
            'balance': round(wallet.balance, 2)
        }
        league_summary.append(report)
        
        print(f"    📊 Bahis: {total_bets} | Win: {total_wins} ({hit_rate:.1f}%)")
        print(f"    💰 PnL: {season_pnl:+.2f} | ROI: {roi:+.1f}% | Bakiye: {wallet.balance:.0f}")
        
        # Sezon raporu kaydet
        with open(league_report_dir / f"{test_season}.json", 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Lig özeti
    total_pnl = sum(s['pnl'] for s in league_summary)
    total_bets = sum(s['bets'] for s in league_summary)
    overall_roi = (total_pnl / (total_bets * STAKE) * 100) if total_bets > 0 else 0
    
    summary = {
        'league': league,
        'seasons_tested': len(league_summary),
        'total_matches': sum(s['matches'] for s in league_summary),
        'total_bets': total_bets,
        'total_wins': sum(s['wins'] for s in league_summary),
        'total_pnl': round(total_pnl, 2),
        'overall_roi': round(overall_roi, 2),
        'final_balance': round(wallet.balance, 2),
        'season_details': league_summary
    }
    
    with open(league_report_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n🏁 [{league}] TAMAMLANDI!")
    print(f"   PnL: {total_pnl:+.2f} TL | ROI: {overall_roi:+.1f}% | Final: {wallet.balance:.0f} TL")
    print(f"   📁 Raporlar: {league_report_dir}")
    print(f"   🤖 Modeller: {league_model_dir}")
    
    return summary


def main():
    print("=" * 70)
    print("   THE ORACLE - TÜM LİGLER İÇİN İZOLE BACKTEST")
    print("=" * 70)
    print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ligler: {ALL_LEAGUES}")
    print()
    print("⚠️  Her lig için 4 AYRÎ model eğitilecek:")
    print("    - Dixon-Coles (1X2)")
    print("    - XGBoost Match Result (1X2)")
    print("    - XGBoost Over/Under (2.5)")
    print("    - XGBoost BTTS")
    print()
    print("⚠️  Modeller birbirinden TAMAMEN bağımsız!")
    print("=" * 70)
    
    output_dir = Path("reports/backtest")
    model_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    start_time = time.time()
    
    for league in ALL_LEAGUES:
        try:
            result = run_league_backtest(league, output_dir, model_dir)
            all_results[league] = result
        except Exception as e:
            print(f"❌ [{league}] KRİTİK HATA: {e}")
            import traceback
            traceback.print_exc()
            all_results[league] = None
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("✅ TÜM BACKTEST TAMAMLANDI!")
    print(f"⏱️ Toplam süre: {elapsed/60:.1f} dakika")
    print(f"📁 Raporlar: {output_dir.absolute()}")
    print(f"🤖 Modeller: {model_dir.absolute()}")
    print("=" * 70)
    
    # Genel özet tablosu
    print("\n📊 LİG BAZINDA ÖZET:")
    print("-" * 50)
    for league, result in all_results.items():
        if result:
            print(f"  {league}: PnL={result['total_pnl']:+.0f} TL | ROI={result['overall_roi']:+.1f}%")
        else:
            print(f"  {league}: HATA")
    
    # Genel özet kaydet
    with open(output_dir / "all_leagues_summary.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'elapsed_minutes': round(elapsed/60, 2),
            'leagues': list(all_results.keys()),
            'results': {k: v for k, v in all_results.items() if v}
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
