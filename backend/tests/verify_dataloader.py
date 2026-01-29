
import sys
from pathlib import Path
from loguru import logger

# Proje kök dizinini path'e ekle
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backend.data.data_loader import DataLoader
from backend.core.config import settings

def test_data_loading():
    logger.info("TEST BAŞLIYOR: Veri Yükleme Doğrulaması")
    
    loader = DataLoader()
    
    # Test edilecek sezonlar (en eski ve en yeni)
    test_seasons = ["2013-2014", "2024-2025"]
    
    for season in test_seasons:
        csv_path = settings.RAW_CSV_DIR / f"{season}.csv"
        logger.info(f"Dosya kontrol ediliyor: {csv_path}")
        
        if not csv_path.exists():
            logger.error(f"HATA: Dosya bulunamadı! {csv_path}")
            continue
            
        try:
            # 1. Hızlı okuma testi
            df = loader.load_csv(csv_path)
            logger.info(f"✓ {season} okundu. Satır sayısı: {len(df)}")
            
            # 2. Temizleme ve Tarih Parse Testi
            df_clean = loader.clean_data(df)
            
            # Tarih formatı kontrolü (YYYY-MM-DD olmalı)
            sample_date = df_clean['date'].iloc[0]
            if len(sample_date) != 10 or sample_date[4] != '-' or sample_date[7] != '-':
                 logger.error(f"HATA: {season} tarih formatı yanlış! Örnek: {sample_date}")
            else:
                 logger.info(f"✓ {season} tarih formatı doğru: {sample_date}")
                 
            # 3. Sütun kontrolü (Time sütunu 2024'te var mı?)
            if season == "2024-2025" and 'Time' in df.columns:
                 logger.info("✓ 2024-2025 dosyasında 'Time' sütunu tespit edildi (Beklenen durum).")
            
        except Exception as e:
            logger.error(f"HATA ({season}): {e}")
            import traceback
            traceback.print_exc()

    logger.info("TEST TAMAMLANDI.")

if __name__ == "__main__":
    test_data_loading()
