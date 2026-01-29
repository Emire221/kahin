"""
The Oracle - Walk-Forward Backtest Script

Bu script, 2013-2014'ten 2024-2025'e kadar sezon sezon walk-forward backtest çalıştırır.
Her sezon için modeli eğitir, bir sonraki sezonda bahis yapar ve raporları kaydeder.

Kullanım:
    python -m backend.walk_forward_backtest

Çıktılar:
    - Konsol: Sezon bazlı özet bilgiler
    - data/logs/: Her sezon için JSON rapor dosyaları
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

from loguru import logger

# Logger yapılandırması
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)

# Proje importları
from backend.core.config import settings
from backend.database.db_manager import DatabaseManager, get_database
from backend.data.data_loader import DataLoader
from backend.simulation.backtest_engine import BacktestEngine, BacktestConfig


# Sezon sırası (raw_csv dosya isimleri)
SEASONS = [
    "2013-2014",
    "2014-2015",
    "2015-2016",
    "2016-2017",
    "2017-2018",
    "2018-2019",
    "2019-2020",
    "2020-2021",
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
]


def reset_database() -> None:
    """Veritabanını tamamen sıfırlar (Tabloları siler ve yeniden oluşturur)."""
    logger.info("Veritabanı sıfırlanıyor...")
    
    db = get_database()
    
    try:
        # Tabloları tamamen sil (DROP)
        with db.get_cursor() as cursor:
            # Foreign key kısıtlamalarını geçici olarak kapat
            cursor.execute("PRAGMA foreign_keys = OFF")
            
            tables = ["matches_history", "predictions", "fixtures", "wallet_simulation"]
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                
            cursor.execute("PRAGMA foreign_keys = ON")
        
        logger.info("✓ Mevcut tablolar silindi")
        
        # Tabloları yeniden oluştur
        db.initialize_db()
        logger.info("✓ Veritabanı yeniden oluşturuldu")
        
    except Exception as e:
        logger.error(f"Veritabanı sıfırlama hatası: {e}")
    
    finally:
        db.close()


def load_season(season: str) -> int:
    """
    Belirli bir sezonu CSV'den veritabanına yükler.
    
    Args:
        season: Sezon adı (örn: "2013-2014")
        
    Returns:
        int: Yüklenen maç sayısı
    """
    csv_file = settings.RAW_CSV_DIR / f"{season}.csv"
    
    if not csv_file.exists():
        logger.error(f"CSV dosyası bulunamadı: {csv_file}")
        return 0
    
    with DataLoader() as loader:
        count = loader.process_and_load(csv_file, season=season)
        
    logger.info(f"✓ {season}: {count} maç yüklendi")
    return count


def run_walk_forward():
    """
    Walk-Forward Backtest ana döngüsü.
    
    Algoritma:
    1. Veritabanını sıfırla
    2. İlk 2 sezonu yükle (minimum eğitim verisi)
    3. Her yeni sezon için:
       a. Model eğit (önceki tüm sezonlarla)
       b. Yeni sezonda bahis yap
       c. Raporu kaydet
       d. Sezonu veritabanına ekle
    """
    print("\n" + "=" * 60)
    print("🔮 THE ORACLE - WALK-FORWARD BACKTEST")
    print("=" * 60)
    print(f"📅 Sezonlar: {SEASONS[0]} → {SEASONS[-1]}")
    print(f"📊 Toplam {len(SEASONS)} sezon\n")
    
    # 1. Veritabanını sıfırla
    reset_database()
    
    # 2. Rapor dizinini oluştur
    reports_dir = settings.LOGS_DIR
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Tüm raporları tutacak liste
    all_reports: List[Dict] = []
    
    # 4. İlk sezonu yükle (minimum 1 sezon eğitim verisi)
    min_train_seasons = 1
    loaded_seasons = []
    
    print(f"\n📥 İlk {min_train_seasons} sezon yükleniyor (eğitim verisi)...")
    for i in range(min_train_seasons):
        count = load_season(SEASONS[i])
        loaded_seasons.append(SEASONS[i])
        
    print(f"✓ Eğitim verisi hazır: {loaded_seasons}")
    
    # 5. Walk-Forward döngüsü
    print("\n" + "-" * 60)
    print("🚀 WALK-FORWARD BACKTEST BAŞLIYOR")
    print("-" * 60)
    
    # Her test sezonu için
    for test_idx in range(min_train_seasons, len(SEASONS)):
        test_season = SEASONS[test_idx]
        train_seasons = loaded_seasons.copy()
        
        print(f"\n📅 Test Sezonu: {test_season}")
        print(f"   Eğitim: {train_seasons}")
        
        # A. Model eğit ve o sezonu test et
        config = BacktestConfig(
            initial_balance=1000.0,
            stake=10.0,
            value_threshold=1.05,
            min_edge=0.03,
            bet_on_value_only=True,
            max_bets_per_day=3
        )
        
        # Önce test sezonunu yükle (sonra modeli eğitip tahmin yapacağız)
        load_season(test_season)
        
        # Backtest engine kullan
        engine = BacktestEngine(config)
        
        try:
            # B. Walk-forward backtest çalıştır
            # Sadece bu sezonu test et (Optimize edilmiş metod)
            season_result = engine.run_season(test_season)
            
            # C. Sezon raporunu kaydet
            report_path = reports_dir / f"backtest_{test_season}.json"
            engine.export_report(report_path)
            
            # D. Sezon sonucunu ekrana yaz
            if season_result:
                print(f"   📊 Sonuçlar:")
                print(f"      Bahis: {season_result.bets_placed}")
                print(f"      Kazanç: {season_result.wins}/{season_result.bets_placed} ({season_result.hit_rate:.1f}%)")
                print(f"      ROI: {season_result.roi:+.2f}%")
                print(f"      Bakiye: {season_result.ending_balance:.2f}")
                print(f"   📁 Rapor: {report_path.name}")
            
            all_reports.append({
                'season': test_season,
                'report_path': str(report_path),
                'bets': season_result.bets_placed if season_result else 0,
                'roi': season_result.roi if season_result else 0
            })
            
        except Exception as e:
            logger.error(f"Backtest hatası ({test_season}): {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            engine.close()
        
        # E. Bu sezonu eğitim listesine ekle
        loaded_seasons.append(test_season)
    
    # 6. Özet rapor
    print("\n" + "=" * 60)
    print("📊 WALK-FORWARD BACKTEST TAMAMLANDI")
    print("=" * 60)
    
    print(f"\n✓ Test Edilen Sezon Sayısı: {len(all_reports)}")
    print("\n📋 Sezon Özeti:")
    print("-" * 40)
    print(f"{'Sezon':<12} {'Bahis':>6} {'ROI':>10}")
    print("-" * 40)
    
    total_roi = 0
    for r in all_reports:
        print(f"{r['season']:<12} {r['bets']:>6} {r['roi']:>+9.2f}%")
        total_roi += r['roi']
    
    print("-" * 40)
    avg_roi = total_roi / len(all_reports) if all_reports else 0
    print(f"{'Ortalama':<12} {'':<6} {avg_roi:>+9.2f}%")
    
    print(f"\n📁 Raporlar: {reports_dir}")
    print("=" * 60 + "\n")
    
    # 7. Ana özet raporunu kaydet
    summary_report = {
        'run_date': datetime.now().isoformat(),
        'seasons_tested': len(all_reports),
        'average_roi': avg_roi,
        'results': all_reports
    }
    
    summary_path = reports_dir / f"walk_forward_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Özet rapor: {summary_path}\n")


if __name__ == "__main__":
    run_walk_forward()
