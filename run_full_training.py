"""
The Oracle - Tam Eğitim ve Backtest Script
==========================================

Bu scripti çalıştırmak için:
    cd c:\Users\ahmet\Desktop\Oracle
    python run_full_training.py

Script sırasıyla:
1. Eski veritabanı verilerini siler
2. Tüm CSV dosyalarını yükler
3. Lig bazlı modelleri eğitir
4. Walk-forward backtest çalıştırır
5. Raporu kaydeder
"""

import sys
from pathlib import Path
from datetime import datetime

# Proje kök dizinini ekle
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from backend.database.db_manager import get_database
from backend.data.data_loader import DataLoader
from backend.training.trainer import ModelTrainer
from backend.simulation.backtest_engine import BacktestEngine, BacktestConfig


def main():
    print("=" * 60)
    print("THE ORACLE - TAM EĞİTİM VE BACKTEST")
    print("=" * 60)
    print(f"Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ========================================
    # ADIM 1: ESKİ VERİLERİ SİL
    # ========================================
    print("\n📁 ADIM 1: Eski veriler siliniyor...")
    db = get_database()
    
    # Tabloları temizle
    db.execute_query("DELETE FROM matches_history")
    db.execute_query("DELETE FROM predictions")
    db.execute_query("DELETE FROM wallet_simulation")
    
    print("✅ Veritabanı temizlendi!\n")
    
    # ========================================
    # ADIM 2: TÜM CSV DOSYALARINI YÜKLE
    # ========================================
    print("📊 ADIM 2: CSV dosyaları yükleniyor...")
    loader = DataLoader()
    
    csv_dir = Path("data/raw_csv")
    csv_files = sorted(csv_dir.glob("*.csv"))
    
    total_matches = 0
    for csv_file in csv_files:
        try:
            count = loader.process_and_load(csv_file, replace_existing=False)
            total_matches += count
            print(f"   ✓ {csv_file.name}: {count} maç")
        except Exception as e:
            print(f"   ✗ {csv_file.name}: HATA - {e}")
    
    print(f"\n✅ Toplam {total_matches} maç yüklendi!\n")
    
    # ========================================
    # ADIM 3: LİG BAZLI MODEL EĞİTİMİ
    # ========================================
    print("🏆 ADIM 3: Lig bazlı modeller eğitiliyor...")
    trainer = ModelTrainer()
    
    # Tier 1 ligleri eğit
    tier1_leagues = ['E0', 'D1', 'I1', 'SP1', 'F1', 'T1', 'N1', 'B1', 'P1']
    
    league_reports = trainer.train_by_league(
        divisions=tier1_leagues,
        tier1_only=True,
        save=True
    )
    
    print("\n📈 Eğitim Sonuçları:")
    for league, report in league_reports.items():
        if 'error' not in report:
            print(f"   {league}: {report['num_matches']} maç, {report['training_time_seconds']:.1f}s")
        else:
            print(f"   {league}: HATA - {report['error']}")
    
    print("\n✅ Model eğitimi tamamlandı!\n")
    
    # ========================================
    # ADIM 4: WALK-FORWARD BACKTEST
    # ========================================
    print("🔄 ADIM 4: Walk-Forward Backtest başlıyor...")
    
    config = BacktestConfig(
        initial_balance=1000.0,
        stake=10.0,
        value_threshold=1.05,
        min_edge=0.03,
        max_bets_per_day=5
    )
    
    engine = BacktestEngine(config)
    
    # En az 2 sezon eğitim verisi ile başla
    report = engine.run_backtest(min_train_seasons=2)
    
    # Raporu yazdır
    print("\n" + "=" * 60)
    engine.print_report(report)
    
    # Raporu kaydet
    report_path = engine.export_report()
    print(f"\n📄 Rapor kaydedildi: {report_path}")
    
    # ========================================
    # ÖZET
    # ========================================
    print("\n" + "=" * 60)
    print("✅ TÜM İŞLEMLER TAMAMLANDI!")
    print("=" * 60)
    print(f"📊 Yüklenen Maç: {total_matches}")
    print(f"🏆 Eğitilen Lig: {len(league_reports)}")
    print(f"💰 Final Bakiye: {report.final_balance:.2f} TL")
    print(f"📈 Toplam ROI: {report.overall_roi:.2f}%")
    print(f"🎯 Hit Rate: {report.overall_hit_rate:.1f}%")
    print(f"⏱️ Bitiş: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
