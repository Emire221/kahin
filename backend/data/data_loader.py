"""
The Oracle - Veri Yükleme ve Temizleme Modülü

Bu modül, football-data.co.uk ve Kaggle formatındaki CSV dosyalarını
okur, temizler, takım isimlerini standardize eder ve veritabanına kaydeder.

Kullanım:
    from backend.data.data_loader import DataLoader
    from backend.core.config import settings
    
    loader = DataLoader()
    
    # Tek CSV dosyası yükle
    df = loader.load_csv("data/raw_csv/E0_2023.csv")
    
    # Temizle ve standardize et
    df_clean = loader.clean_data(df)
    df_standard = loader.standardize_team_names(df_clean)
    
    # Veritabanına kaydet
    count = loader.load_to_database(df_standard)
    
    # Veya hepsini tek seferde:
    count = loader.process_and_load("data/raw_csv/E0_2023.csv", season="2023-2024")

Desteklenen Formatlar:
    - football-data.co.uk CSV formatı
    - Kaggle Premier League CSV formatı
"""

import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union

import pandas as pd
import numpy as np
from loguru import logger

from backend.core.config import settings
from backend.database.db_manager import DatabaseManager, get_database


class DataLoader:
    """
    CSV veri yükleme ve işleme sınıfı.
    
    Bu sınıf, ham CSV dosyalarını okur, verileri temizler,
    takım isimlerini standardize eder ve veritabanına yükler.
    
    Attributes:
        db_manager (DatabaseManager): Veritabanı yöneticisi
        team_mapping (Dict): Takım isim eşleştirme sözlüğü
        reverse_mapping (Dict): Ters takım eşleştirmesi
        
    Example:
        >>> loader = DataLoader()
        >>> count = loader.process_all_csv_files()
        >>> print(f"{count} maç yüklendi")
    """
    
    # football-data.co.uk sütun eşleştirmesi
    COLUMN_MAPPING = {
        # Temel sütunlar
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'fthg',
        'FTAG': 'ftag',
        'FTR': 'result',
        'Div': 'division',  # Lig kodu
        'Referee': 'referee',  # Hakem
        
        # Alternatif isimler
        'Home': 'home_team',
        'Away': 'away_team',
        'HG': 'fthg',
        'AG': 'ftag',
        'Res': 'result',
        
        # İlk yarı skorları
        'HTHG': 'hthg',
        'HTAG': 'htag',
        'HTR': 'htr',
        
        # İstatistik sütunları
        'HS': 'hs',
        'AS': 'as',
        'HST': 'hst',
        'AST': 'ast',
        'HC': 'hc',
        'AC': 'ac',
        'HF': 'hf',
        'AF': 'af',
        'HY': 'hy',
        'AY': 'ay',
        'HR': 'hr',
        'AR': 'ar',
        
        # Bahis oranları - Temel
        'B365H': 'home_odds',
        'B365D': 'draw_odds',
        'B365A': 'away_odds',
        'BbAvH': 'home_odds',
        'BbAvD': 'draw_odds',
        'BbAvA': 'away_odds',
        
        # Genişletilmiş oranlar (stats JSON'a)
        'AvgH': 'avgh',
        'AvgD': 'avgd',
        'AvgA': 'avga',
        'MaxH': 'maxh',
        'MaxD': 'maxd',
        'MaxA': 'maxa',
        'B365CH': 'b365ch',
        'B365CD': 'b365cd',
        'B365CA': 'b365ca',
        'B365>2.5': 'b365>2.5',
        'B365<2.5': 'b365<2.5',
        'AHh': 'ahh',
        'B365AHH': 'b365ahh',
        'B365AHA': 'b365aha',
    }
    
    # Tier 1 Ligleri (Ana Ligler) - Yüksek kalite veri, bahis hedefi
    TIER_1_LEAGUES = {'E0', 'D1', 'I1', 'SP1', 'F1', 'T1', 'N1', 'B1', 'P1'}
    
    # Tier 2 Ligleri (Alt Ligler) - Geçmiş veri, terfi/küme düşme takibi için
    TIER_2_LEAGUES = {'E1', 'D2', 'I2', 'SP2', 'F2'}
    
    # Tarih formatları (football-data.co.uk çeşitli formatlar kullanır)
    DATE_FORMATS = [
        '%d/%m/%Y',      # 25/08/2023
        '%d/%m/%y',      # 25/08/23
        '%Y-%m-%d',      # 2023-08-25
        '%d-%m-%Y',      # 25-08-2023
        '%m/%d/%Y',      # 08/25/2023
    ]
    
    def __init__(
        self, 
        db_manager: Optional[DatabaseManager] = None,
        team_mapping_path: Optional[Path] = None
    ) -> None:
        """
        DataLoader sınıfını başlatır.
        
        Args:
            db_manager: Veritabanı yöneticisi (None ise otomatik oluşturulur)
            team_mapping_path: Takım eşleştirme JSON dosyası yolu
        """
        self.db_manager = db_manager
        self._owns_db_manager = db_manager is None
        
        # Takım eşleştirmesini yükle
        mapping_path = team_mapping_path or settings.TEAM_MAPPING_FILE
        self.team_mapping, self.reverse_mapping = self._load_team_mapping(mapping_path)
        
        logger.info("DataLoader başlatıldı")
    
    def _load_team_mapping(
        self, 
        mapping_path: Path
    ) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Takım eşleştirme dosyasını yükler.
        
        Args:
            mapping_path: JSON dosyası yolu
            
        Returns:
            Tuple: (normal mapping, reverse mapping)
        """
        try:
            if not mapping_path.exists():
                logger.warning(f"Takım eşleştirme dosyası bulunamadı: {mapping_path}")
                return {}, {}
            
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            # Ters eşleştirme oluştur
            reverse = {}
            for standard_name, alternatives in mapping.items():
                reverse[standard_name] = standard_name
                for alt in alternatives:
                    reverse[alt] = standard_name
            
            logger.info(f"Takım eşleştirmesi yüklendi: {len(mapping)} takım")
            return mapping, reverse
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse hatası: {e}")
            return {}, {}
        except Exception as e:
            logger.error(f"Takım eşleştirme yükleme hatası: {e}")
            return {}, {}
    
    def _get_db_manager(self) -> DatabaseManager:
        """
        Veritabanı yöneticisini döndürür.
        Yoksa yeni bir tane oluşturur.
        """
        if self.db_manager is None:
            self.db_manager = get_database()
            self._owns_db_manager = True
        return self.db_manager
    
    def parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        Dosya adından lig kodu ve sezon bilgisi çıkarır.
        
        Format: KOD_SEZON.csv (örn: E0_2023-2024.csv, T1_2122.csv)
        
        Args:
            filename: CSV dosya adı
            
        Returns:
            Dict: {'division': str, 'season': str, 'tier': int}
            
        Example:
            >>> loader.parse_filename("E0_2023-2024.csv")
            {'division': 'E0', 'season': '2023-2024', 'tier': 1}
            >>> loader.parse_filename("D2_2122.csv")
            {'division': 'D2', 'season': '2021-2022', 'tier': 2}
        """
        result = {'division': None, 'season': None, 'tier': 1}
        
        # Dosya adından uzantıyı çıkar
        name = Path(filename).stem
        
        # Pattern 1: KOD_YYYY-YYYY (örn: E0_2023-2024)
        match = re.match(r'^([A-Z]+\d?)_(\d{4})-(\d{4})$', name)
        if match:
            result['division'] = match.group(1)
            result['season'] = f"{match.group(2)}-{match.group(3)}"
            result['tier'] = self.get_tier(result['division'])
            return result
        
        # Pattern 2: KOD_YYZZ (örn: E0_2324 -> 2023-2024)
        match = re.match(r'^([A-Z]+\d?)_(\d{2})(\d{2})$', name)
        if match:
            result['division'] = match.group(1)
            year1 = int(match.group(2))
            year2 = int(match.group(3))
            
            # Yüzyılı belirle (90-99: 1990s, 00-24: 2000s)
            century1 = 1900 if year1 >= 90 else 2000
            century2 = 1900 if year2 >= 90 else 2000
            
            result['season'] = f"{century1 + year1}-{century2 + year2}"
            result['tier'] = self.get_tier(result['division'])
            return result
        
        # Pattern 3: KOD_YYYY (tek yıl, örn: E0_2023)
        match = re.match(r'^([A-Z]+\d?)_(\d{4})$', name)
        if match:
            result['division'] = match.group(1)
            year = int(match.group(2))
            result['season'] = f"{year}-{year + 1}"
            result['tier'] = self.get_tier(result['division'])
            return result
        
        logger.warning(f"Dosya adı parse edilemedi: {filename}")
        return result
    
    def get_tier(self, division: str) -> int:
        """
        Lig kodundan tier seviyesini döndürür.
        
        Args:
            division: Lig kodu (E0, T1, D2 vb.)
            
        Returns:
            int: 1 (Tier 1 - Ana Lig) veya 2 (Tier 2 - Alt Lig)
        """
        if division in self.TIER_1_LEAGUES:
            return 1
        elif division in self.TIER_2_LEAGUES:
            return 2
        else:
            # Bilinmeyen ligler için varsayılan
            logger.debug(f"Bilinmeyen lig kodu, Tier 1 varsayıldı: {division}")
            return 1
    
    def load_csv(
        self, 
        file_path: Union[str, Path],
        encoding: Optional[str] = None
    ) -> pd.DataFrame:
        """
        CSV dosyasını okur ve DataFrame döndürür.
        
        Args:
            file_path: CSV dosyasının yolu
            encoding: Dosya kodlaması (None ise otomatik algılanır)
            
        Returns:
            pd.DataFrame: Okunan veri
            
        Raises:
            FileNotFoundError: Dosya bulunamazsa
            pd.errors.ParserError: CSV parse hatası
            
        Example:
            >>> df = loader.load_csv("data/raw_csv/E0_2023.csv")
            >>> print(df.shape)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {file_path}")
        
        # Kodlama listesi (sırayla dene)
        encodings = [encoding] if encoding else ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        last_error = None
        for enc in encodings:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=enc,
                    on_bad_lines='skip',  # Hatalı satırları atla
                    low_memory=False
                )
                
                logger.info(f"CSV yüklendi: {file_path.name} ({len(df)} satır, encoding: {enc})")
                return df
                
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except Exception as e:
                logger.error(f"CSV okuma hatası ({enc}): {e}")
                last_error = e
                continue
        
        raise ValueError(f"CSV dosyası okunamadı: {file_path}. Son hata: {last_error}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sütun isimlerini standart formata dönüştürür.
        
        Args:
            df: Orijinal DataFrame
            
        Returns:
            pd.DataFrame: Standart sütun isimli DataFrame
        """
        df = df.copy()
        
        # Sütun isimlerini eşleştir
        rename_dict = {}
        for col in df.columns:
            col_stripped = col.strip()
            if col_stripped in self.COLUMN_MAPPING:
                rename_dict[col] = self.COLUMN_MAPPING[col_stripped]
        
        df = df.rename(columns=rename_dict)
        
        return df
    
    def _parse_date(self, date_value: Any) -> Optional[str]:
        """
        Tarih değerini standart formata (YYYY-MM-DD) dönüştürür.
        
        Args:
            date_value: Tarih değeri (string veya datetime)
            
        Returns:
            str: YYYY-MM-DD formatında tarih veya None
        """
        if pd.isna(date_value):
            return None
        
        # Zaten datetime ise
        if isinstance(date_value, (datetime, pd.Timestamp)):
            return date_value.strftime('%Y-%m-%d')
        
        date_str = str(date_value).strip()
        
        # Farklı formatları dene
        for fmt in self.DATE_FORMATS:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        logger.warning(f"Tarih parse edilemedi: {date_str}")
        return None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Veriyi temizler ve doğrular.
        
        İşlemler:
            1. Sütun isimlerini standardize et
            2. Tarihleri parse et
            3. Boş/geçersiz satırları sil
            4. Veri tiplerini düzelt
            5. Result değerlerini kontrol et
        
        Args:
            df: Ham DataFrame
            
        Returns:
            pd.DataFrame: Temizlenmiş DataFrame
            
        Example:
            >>> df_raw = loader.load_csv("data.csv")
            >>> df_clean = loader.clean_data(df_raw)
        """
        logger.info(f"Veri temizleme başladı: {len(df)} satır")
        
        # Sütun isimlerini standardize et
        df = self._standardize_columns(df)
        
        # Gerekli sütunları kontrol et
        required_cols = ['date', 'home_team', 'away_team', 'fthg', 'ftag', 'result']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Eksik sütunlar: {missing}. Mevcut: {list(df.columns)}")
        
        # Tarihleri parse et
        df['date'] = df['date'].apply(self._parse_date)
        
        # Boş tarihleri sil
        initial_count = len(df)
        df = df.dropna(subset=['date'])
        if len(df) < initial_count:
            logger.warning(f"{initial_count - len(df)} satır geçersiz tarih nedeniyle silindi")
        
        # Boş takım isimlerini sil
        df = df.dropna(subset=['home_team', 'away_team'])
        
        # Takım isimlerini temizle (whitespace)
        df['home_team'] = df['home_team'].astype(str).str.strip()
        df['away_team'] = df['away_team'].astype(str).str.strip()
        
        # Gol değerlerini sayıya çevir
        df['fthg'] = pd.to_numeric(df['fthg'], errors='coerce').fillna(0).astype(int)
        df['ftag'] = pd.to_numeric(df['ftag'], errors='coerce').fillna(0).astype(int)
        
        # Result değerlerini kontrol et ve düzelt
        df['result'] = df['result'].astype(str).str.strip().str.upper()
        
        # Geçersiz result değerlerini düzelt (skordan hesapla)
        def calculate_result(row):
            if row['result'] in ['H', 'D', 'A']:
                return row['result']
            
            if row['fthg'] > row['ftag']:
                return 'H'
            elif row['fthg'] < row['ftag']:
                return 'A'
            else:
                return 'D'
        
        df['result'] = df.apply(calculate_result, axis=1)
        
        # İstatistik sütunlarını sayıya çevir
        stat_cols = ['hs', 'as', 'hst', 'ast', 'hc', 'ac', 'hf', 'af', 'hy', 'ay', 'hr', 'ar']
        for col in stat_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Bahis oranlarını sayıya çevir
        odds_cols = ['home_odds', 'draw_odds', 'away_odds']
        for col in odds_cols:
            if col in df.columns and isinstance(df[col], pd.Series):
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Negatif değerleri temizle
        df = df[(df['fthg'] >= 0) & (df['ftag'] >= 0)]
        
        # Aykırı değerleri temizle (10'dan fazla gol şüpheli)
        df = df[(df['fthg'] <= 10) & (df['ftag'] <= 10)]
        
        logger.info(f"Veri temizleme tamamlandı: {len(df)} satır kaldı")
        
        return df
    
    def standardize_team_names(
        self, 
        df: pd.DataFrame,
        home_col: str = 'home_team',
        away_col: str = 'away_team'
    ) -> pd.DataFrame:
        """
        Takım isimlerini standart formata dönüştürür.
        
        team_mapping.json dosyasındaki eşleştirmeyi kullanarak
        farklı kaynaklardaki takım isimlerini birleştirir.
        
        Args:
            df: DataFrame
            home_col: Ev sahibi takım sütunu
            away_col: Deplasman takım sütunu
            
        Returns:
            pd.DataFrame: Standardize edilmiş DataFrame
            
        Example:
            >>> df['home_team'] = ['Man Utd', 'Arsenal']
            >>> df = loader.standardize_team_names(df)
            >>> print(df['home_team'].tolist())
            ['Man United', 'Arsenal']
        """
        if not self.reverse_mapping:
            logger.warning("Takım eşleştirmesi boş, standardizasyon atlanıyor")
            return df
        
        df = df.copy()
        
        def standardize_name(name: str) -> str:
            """Tek bir takım ismini standardize eder"""
            if pd.isna(name):
                return name
            
            name = str(name).strip()
            
            # Direkt eşleşme
            if name in self.reverse_mapping:
                return self.reverse_mapping[name]
            
            # Büyük/küçük harf duyarsız arama
            name_lower = name.lower()
            for key, value in self.reverse_mapping.items():
                if key.lower() == name_lower:
                    return value
            
            # Eşleşme bulunamadı, orijinal ismi döndür
            return name
        
        # Sütunları standardize et
        original_home = df[home_col].copy()
        original_away = df[away_col].copy()
        
        df[home_col] = df[home_col].apply(standardize_name)
        df[away_col] = df[away_col].apply(standardize_name)
        
        # Değişen isimleri logla
        changed_home = (original_home != df[home_col]).sum()
        changed_away = (original_away != df[away_col]).sum()
        
        if changed_home > 0 or changed_away > 0:
            logger.info(f"Takım isimleri standardize edildi: {changed_home} ev, {changed_away} deplasman")
        
        return df
    
    def add_season_column(
        self, 
        df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        Tarihten sezon bilgisi hesaplar ve ekler.
        
        Premier Lig sezonu genelde Ağustos-Mayıs arasıdır.
        Ağustos-Aralık arası -> mevcut yıl sezonu
        Ocak-Mayıs arası -> önceki yıl sezonu
        
        Args:
            df: DataFrame
            date_col: Tarih sütunu
            
        Returns:
            pd.DataFrame: Sezon sütunu eklenmiş DataFrame
        """
        df = df.copy()
        
        def get_season(date_str: str) -> str:
            """Tarihten sezon hesaplar"""
            try:
                date = datetime.strptime(date_str, '%Y-%m-%d')
                
                # Ağustos ve sonrası: mevcut yıl sezonu
                # Temmuz ve öncesi: önceki yıl sezonu
                if date.month >= 7:  # Temmuz'dan itibaren yeni sezon
                    start_year = date.year
                else:
                    start_year = date.year - 1
                
                end_year = start_year + 1
                return f"{start_year}-{end_year}"
                
            except (ValueError, TypeError):
                return None
        
        df['season'] = df[date_col].apply(get_season)
        
        # Sezon dağılımını logla
        if 'season' in df.columns:
            season_counts = df['season'].value_counts().to_dict()
            logger.debug(f"Sezon dağılımı: {season_counts}")
        
        return df
    
    def load_to_database(
        self, 
        df: pd.DataFrame,
        replace_existing: bool = False
    ) -> int:
        """
        DataFrame'i veritabanına yükler.
        
        Args:
            df: Yüklenecek DataFrame
            replace_existing: Var olan kayıtları değiştir
            
        Returns:
            int: Eklenen kayıt sayısı
            
        Example:
            >>> df = loader.load_csv("data.csv")
            >>> df = loader.clean_data(df)
            >>> count = loader.load_to_database(df)
        """
        db = self._get_db_manager()
        
        # Veritabanını hazırla
        db.initialize_db()
        
        # Veriyi yükle
        count = db.bulk_insert_matches(df, replace_existing=replace_existing)
        
        return count
    
    def process_and_load(
        self,
        file_path: Union[str, Path],
        season: Optional[str] = None,
        division: Optional[str] = None,
        tier: Optional[int] = None,
        replace_existing: bool = False
    ) -> int:
        """
        CSV dosyasını okur, temizler ve veritabanına yükler.
        Tek adımda tüm işlemleri yapar.
        
        Division, tier ve season bilgisi dosya adından otomatik çıkarılır.
        Manuel değer verilirse o kullanılır.
        
        Args:
            file_path: CSV dosyası yolu
            season: Sezon bilgisi (None ise dosya adından çıkarılır)
            division: Lig kodu (None ise dosya adından çıkarılır)
            tier: Lig seviyesi (None ise division'dan hesaplanır)
            replace_existing: Var olan kayıtları değiştir
            
        Returns:
            int: Yüklenen kayıt sayısı
            
        Example:
            >>> count = loader.process_and_load(
            ...     "data/raw_csv/E0_2023-2024.csv"
            ... )  # division=E0, tier=1, season=2023-2024 otomatik
        """
        file_path = Path(file_path)
        logger.info(f"İşlem başlıyor: {file_path}")
        
        try:
            # Dosya adından metadata çıkar
            file_info = self.parse_filename(file_path.name)
            
            # Manuel değer verilmemişse dosya adından al
            if division is None:
                division = file_info.get('division')
            if tier is None:
                tier = file_info.get('tier', 1)
            if season is None:
                season = file_info.get('season')
            
            logger.debug(f"Metadata: division={division}, tier={tier}, season={season}")
            
            # 1. CSV'yi oku
            df = self.load_csv(file_path)
            
            # 2. Veriyi temizle
            df = self.clean_data(df)
            
            # 3. Takım isimlerini standardize et
            df = self.standardize_team_names(df)
            
            # 4. Division ve Tier bilgilerini ekle
            if division:
                df['division'] = division
            if tier:
                df['tier'] = tier
            
            # 5. Sezon ekle
            if season:
                df['season'] = season
            else:
                df = self.add_season_column(df)
            
            # 6. Referee bilgisini işle (CSV'de varsa)
            if 'referee' not in df.columns:
                df['referee'] = None
            
            # 7. Veritabanına yükle
            count = self.load_to_database(df, replace_existing=replace_existing)
            
            logger.info(f"İşlem tamamlandı: {count} kayıt yüklendi ({division}/{tier}/{season})")
            return count
            
        except Exception as e:
            logger.error(f"process_and_load hatası: {e}")
            raise
    
    def process_all_csv_files(
        self,
        csv_dir: Optional[Path] = None,
        pattern: str = "*.csv",
        replace_existing: bool = False
    ) -> int:
        """
        Bir dizindeki tüm CSV dosyalarını işler ve yükler.
        
        Args:
            csv_dir: CSV dosyalarının bulunduğu dizin
            pattern: Dosya deseni (glob pattern)
            replace_existing: Var olan kayıtları değiştir
            
        Returns:
            int: Toplam yüklenen kayıt sayısı
            
        Example:
            >>> count = loader.process_all_csv_files()
            >>> print(f"Toplam {count} maç yüklendi")
        """
        csv_dir = csv_dir or settings.RAW_CSV_DIR
        csv_dir = Path(csv_dir)
        
        if not csv_dir.exists():
            logger.warning(f"CSV dizini bulunamadı: {csv_dir}")
            return 0
        
        csv_files = list(csv_dir.glob(pattern))
        
        if not csv_files:
            logger.warning(f"CSV dosyası bulunamadı: {csv_dir}/{pattern}")
            return 0
        
        logger.info(f"{len(csv_files)} CSV dosyası bulundu")
        
        total_count = 0
        successful = 0
        failed = 0
        
        for csv_file in sorted(csv_files):
            try:
                count = self.process_and_load(csv_file, replace_existing=replace_existing)
                total_count += count
                successful += 1
                
            except Exception as e:
                logger.error(f"Dosya işleme hatası ({csv_file.name}): {e}")
                failed += 1
                continue
        
        logger.info(
            f"Toplu yükleme tamamlandı: "
            f"{successful} başarılı, {failed} başarısız, "
            f"toplam {total_count} kayıt"
        )
        
        return total_count
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Veritabanındaki veri özetini döndürür.
        
        Returns:
            Dict: İstatistikler
        """
        db = self._get_db_manager()
        
        # Tablo var mı kontrol et
        if not db.table_exists('matches_history'):
            return {'status': 'empty', 'total_matches': 0}
        
        # Maç sayısı
        total = db.get_match_count()
        
        # Sezon dağılımı
        seasons = db.execute_query("""
            SELECT season, COUNT(*) as count 
            FROM matches_history 
            GROUP BY season 
            ORDER BY season
        """)
        
        # Takım sayısı
        teams = db.execute_query("""
            SELECT COUNT(DISTINCT home_team) as home_teams,
                   COUNT(DISTINCT away_team) as away_teams
            FROM matches_history
        """)
        
        # Tarih aralığı
        date_range = db.execute_query("""
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM matches_history
        """)
        
        return {
            'status': 'ok',
            'total_matches': total,
            'seasons': {s['season']: s['count'] for s in seasons},
            'unique_teams': teams[0]['home_teams'] if teams else 0,
            'date_range': {
                'start': date_range[0]['min_date'] if date_range else None,
                'end': date_range[0]['max_date'] if date_range else None
            }
        }
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Veritabanındaki verilerin tutarlılığını kontrol eder.
        
        Returns:
            Dict: Doğrulama sonuçları
        """
        db = self._get_db_manager()
        
        issues = []
        
        # Boş takım isimleri
        empty_teams = db.execute_query("""
            SELECT COUNT(*) as count FROM matches_history
            WHERE home_team = '' OR away_team = ''
        """)
        if empty_teams and empty_teams[0]['count'] > 0:
            issues.append(f"Boş takım ismi: {empty_teams[0]['count']} kayıt")
        
        # Geçersiz sonuçlar
        invalid_results = db.execute_query("""
            SELECT COUNT(*) as count FROM matches_history
            WHERE result NOT IN ('H', 'D', 'A')
        """)
        if invalid_results and invalid_results[0]['count'] > 0:
            issues.append(f"Geçersiz sonuç: {invalid_results[0]['count']} kayıt")
        
        # Negatif goller
        negative_goals = db.execute_query("""
            SELECT COUNT(*) as count FROM matches_history
            WHERE fthg < 0 OR ftag < 0
        """)
        if negative_goals and negative_goals[0]['count'] > 0:
            issues.append(f"Negatif gol: {negative_goals[0]['count']} kayıt")
        
        # Tutarsız sonuçlar (skor ile result uyuşmuyor)
        inconsistent = db.execute_query("""
            SELECT COUNT(*) as count FROM matches_history
            WHERE (fthg > ftag AND result != 'H')
               OR (fthg < ftag AND result != 'A')
               OR (fthg = ftag AND result != 'D')
        """)
        if inconsistent and inconsistent[0]['count'] > 0:
            issues.append(f"Tutarsız sonuç: {inconsistent[0]['count']} kayıt")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    def close(self) -> None:
        """Kaynakları temizler"""
        if self._owns_db_manager and self.db_manager:
            self.db_manager.close()
            self.db_manager = None
    
    def __enter__(self) -> "DataLoader":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def quick_load(file_path: Union[str, Path], season: Optional[str] = None) -> int:
    """
    Hızlı CSV yükleme fonksiyonu.
    
    Args:
        file_path: CSV dosya yolu
        season: Sezon bilgisi
        
    Returns:
        int: Yüklenen kayıt sayısı
        
    Example:
        >>> from backend.data.data_loader import quick_load
        >>> count = quick_load("data/raw_csv/E0_2023.csv", "2023-2024")
    """
    with DataLoader() as loader:
        return loader.process_and_load(file_path, season=season)


def load_all_seasons() -> int:
    """
    Tüm sezonların CSV dosyalarını yükler.
    
    Returns:
        int: Toplam yüklenen kayıt sayısı
    """
    with DataLoader() as loader:
        return loader.process_all_csv_files()


def get_summary() -> Dict[str, Any]:
    """
    Veritabanı veri özetini döndürür.
    
    Returns:
        Dict: İstatistikler
    """
    with DataLoader() as loader:
        return loader.get_data_summary()
