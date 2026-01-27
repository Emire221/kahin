"""
The Oracle - Merkezi Konfigürasyon Modülü

Bu modül, projenin tüm yol (path), veritabanı ve API ayarlarını
merkezi bir noktada yönetir. Tüm modüller bu ayarları buradan import eder.

Kullanım:
    from backend.core.config import settings
    
    db_path = settings.DATABASE_PATH
    raw_csv_dir = settings.RAW_CSV_DIR
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from functools import lru_cache

from pydantic_settings import BaseSettings


# Ana proje dizinini hesapla (Bu dosyanın 3 üst klasörü)
_BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Proje konfigürasyon sınıfı.
    
    Tüm ayarlar burada tanımlanır ve çevresel değişkenlerden
    veya varsayılan değerlerden okunur.
    """
    
    # ========== PROJE TEMEL AYARLARI ==========
    PROJECT_NAME: str = "The Oracle"
    PROJECT_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # ========== DİZİN YOLLARI ==========
    # Tüm path'ler doğrudan hesaplanıyor (Pydantic validation sorununu çözer)
    BASE_DIR: Path = _BASE_DIR
    BACKEND_DIR: Path = _BASE_DIR / "backend"
    DATA_DIR: Path = _BASE_DIR / "data"
    RAW_CSV_DIR: Path = _BASE_DIR / "data" / "raw_csv"
    LOGS_DIR: Path = _BASE_DIR / "data" / "logs"
    MODELS_DIR: Path = _BASE_DIR / "backend" / "models"
    
    # ========== VERİTABANI AYARLARI ==========
    DATABASE_NAME: str = "oracle.db"
    DATABASE_PATH: Path = _BASE_DIR / "data" / "oracle.db"
    
    # SQLite bağlantı ayarları
    DB_TIMEOUT: int = 30  # Saniye cinsinden bağlantı zaman aşımı
    DB_CHECK_SAME_THREAD: bool = False  # Multi-thread erişim için
    
    # ========== TAKİM EŞLEŞTİRME ==========
    TEAM_MAPPING_FILE: Path = _BASE_DIR / "backend" / "core" / "team_mapping.json"
    
    # ========== API AYARLARI ==========
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    API_PREFIX: str = "/api"
    
    # ========== LOGGING AYARLARI ==========
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    
    # ========== VERİ KAYNAĞI AYARLARI ==========
    # football-data.co.uk formatı için sütun adları
    CSV_DATE_FORMAT: str = "%d/%m/%Y"
    CSV_ENCODING: str = "utf-8"
    
    # Kullanılacak CSV sütunları (football-data.co.uk formatı)
    CSV_REQUIRED_COLUMNS: List[str] = [
        "Date",      # Maç tarihi
        "HomeTeam",  # Ev sahibi takım
        "AwayTeam",  # Deplasman takımı
        "FTHG",      # Full Time Home Goals (Ev sahibi gol)
        "FTAG",      # Full Time Away Goals (Deplasman gol)
        "FTR",       # Full Time Result (H/D/A)
    ]
    
    # Opsiyonel CSV sütunları (varsa çekilir)
    CSV_OPTIONAL_COLUMNS: List[str] = [
        "HS",        # Home Shots (Ev sahibi şut)
        "AS",        # Away Shots (Deplasman şut)
        "HST",       # Home Shots on Target (Ev sahibi isabetli şut)
        "AST",       # Away Shots on Target (Deplasman isabetli şut)
        "HC",        # Home Corners (Ev sahibi korner)
        "AC",        # Away Corners (Deplasman korner)
        "HF",        # Home Fouls (Ev sahibi faul)
        "AF",        # Away Fouls (Deplasman faul)
        "HY",        # Home Yellow Cards (Ev sahibi sarı kart)
        "AY",        # Away Yellow Cards (Deplasman sarı kart)
        "HR",        # Home Red Cards (Ev sahibi kırmızı kart)
        "AR",        # Away Red Cards (Deplasman kırmızı kart)
        "B365H",     # Bet365 Home Odds (Ev sahibi oranı)
        "B365D",     # Bet365 Draw Odds (Beraberlik oranı)
        "B365A",     # Bet365 Away Odds (Deplasman oranı)
    ]
    
    # ========== BACKTEST AYARLARI ==========
    INITIAL_BANKROLL: float = 1000.0  # Başlangıç kasası
    FLAT_STAKE: float = 10.0  # Sabit bahis miktarı
    VALUE_THRESHOLD: float = 1.05  # Value bet eşik değeri (EV > 1.05)
    
    # ========== WALK-FORWARD AYARLARI ==========
    TRAIN_START_YEAR: int = 2014  # Eğitim başlangıç yılı
    TRAIN_END_YEAR: int = 2018    # İlk eğitim bitiş yılı
    
    class Config:
        """Pydantic konfigürasyonu"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def ensure_directories(self) -> None:
        """
        Gerekli dizinlerin var olduğundan emin olur.
        Yoksa oluşturur.
        """
        directories = [
            self.DATA_DIR,
            self.RAW_CSV_DIR,
            self.LOGS_DIR,
            self.MODELS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_team_mapping(self) -> Dict[str, List[str]]:
        """
        Takım eşleştirme JSON dosyasını yükler.
        
        Returns:
            Dict[str, List[str]]: Standart isim -> Alternatif isimler sözlüğü
        
        Raises:
            FileNotFoundError: Dosya bulunamazsa
            json.JSONDecodeError: JSON parse hatası
        """
        if not self.TEAM_MAPPING_FILE.exists():
            raise FileNotFoundError(
                f"Takım eşleştirme dosyası bulunamadı: {self.TEAM_MAPPING_FILE}"
            )
        
        with open(self.TEAM_MAPPING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def get_reverse_team_mapping(self) -> Dict[str, str]:
        """
        Ters takım eşleştirmesi döndürür.
        Alternatif isim -> Standart isim formatında.
        
        Returns:
            Dict[str, str]: Alternatif isim -> Standart isim sözlüğü
        """
        mapping = self.load_team_mapping()
        reverse_mapping: Dict[str, str] = {}
        
        for standard_name, alternatives in mapping.items():
            # Standart ismi de ekle (kendi kendine eşleşir)
            reverse_mapping[standard_name] = standard_name
            
            for alt_name in alternatives:
                reverse_mapping[alt_name] = standard_name
        
        return reverse_mapping


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern ile settings nesnesini döndürür.
    
    Returns:
        Settings: Konfigürasyon nesnesi
    """
    return Settings()


# Global settings instance
settings = get_settings()
