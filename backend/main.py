"""
The Oracle - Ana Giriş Noktası (Entry Point)

Bu dosya, projenin ana çalıştırılabilir dosyasıdır.
Hem CLI komutlarını hem de FastAPI sunucusunu buradan başlatabilirsiniz.

Kullanım:
    # Veritabanını başlat
    python -m backend.main init-db
    
    # CSV dosyalarını yükle
    python -m backend.main load-data
    
    # API sunucusunu başlat
    python -m backend.main serve
    
    # Veri özetini göster
    python -m backend.main summary
"""

import sys
import argparse
from pathlib import Path

from loguru import logger

# Logger yapılandırması
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)


def init_database() -> None:
    """Veritabanını oluşturur ve tabloları hazırlar."""
    from backend.core.config import settings
    from backend.database.db_manager import DatabaseManager
    
    logger.info("Veritabanı başlatılıyor...")
    
    # Dizinleri oluştur
    settings.ensure_directories()
    
    # Veritabanını başlat
    with DatabaseManager(settings.DATABASE_PATH) as db:
        db.initialize_db()
    
    logger.info(f"Veritabanı hazır: {settings.DATABASE_PATH}")


def load_data(csv_path: str = None) -> None:
    """CSV dosyalarını veritabanına yükler."""
    from backend.core.config import settings
    from backend.data.data_loader import DataLoader
    
    logger.info("Veri yükleme başlıyor...")
    
    with DataLoader() as loader:
        if csv_path:
            # Tek dosya yükle
            count = loader.process_and_load(csv_path)
            logger.info(f"{count} kayıt yüklendi: {csv_path}")
        else:
            # Tüm CSV dosyalarını yükle
            count = loader.process_all_csv_files()
            logger.info(f"Toplam {count} kayıt yüklendi")


def show_summary() -> None:
    """Veritabanı özetini gösterir."""
    from backend.data.data_loader import DataLoader
    
    with DataLoader() as loader:
        summary = loader.get_data_summary()
        
        print("\n" + "=" * 50)
        print("THE ORACLE - VERİ ÖZETİ")
        print("=" * 50)
        
        if summary['status'] == 'empty':
            print("Veritabanı boş. Önce veri yükleyin.")
            return
        
        print(f"Toplam Maç: {summary['total_matches']}")
        print(f"Takım Sayısı: {summary['unique_teams']}")
        print(f"Tarih Aralığı: {summary['date_range']['start']} - {summary['date_range']['end']}")
        
        print("\nSezon Dağılımı:")
        for season, count in summary['seasons'].items():
            print(f"  {season}: {count} maç")
        
        # Veri doğrulama
        validation = loader.validate_data()
        if not validation['valid']:
            print("\n⚠️  Veri Sorunları:")
            for issue in validation['issues']:
                print(f"  - {issue}")
        else:
            print("\n✓ Veri tutarlılığı OK")
        
        print("=" * 50 + "\n")


def serve_api() -> None:
    """FastAPI sunucusunu başlatır."""
    import uvicorn
    from backend.core.config import settings
    
    logger.info(f"API sunucusu başlatılıyor: http://{settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "backend.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )


def main():
    """Ana CLI fonksiyonu."""
    parser = argparse.ArgumentParser(
        description="The Oracle - Premier Lig Tahmin Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Komutlar')
    
    # init-db komutu
    subparsers.add_parser('init-db', help='Veritabanını başlat')
    
    # load-data komutu
    load_parser = subparsers.add_parser('load-data', help='CSV verilerini yükle')
    load_parser.add_argument(
        '--file', '-f',
        type=str,
        help='Yüklenecek CSV dosyası (belirtilmezse tüm dosyalar yüklenir)'
    )
    
    # summary komutu
    subparsers.add_parser('summary', help='Veri özetini göster')
    
    # serve komutu
    subparsers.add_parser('serve', help='API sunucusunu başlat')
    
    args = parser.parse_args()
    
    if args.command == 'init-db':
        init_database()
    elif args.command == 'load-data':
        init_database()  # Önce veritabanını hazırla
        load_data(args.file if hasattr(args, 'file') else None)
    elif args.command == 'summary':
        show_summary()
    elif args.command == 'serve':
        serve_api()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
