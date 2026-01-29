"""
The Oracle - Veritabanı Yönetim Modülü

Bu modül, SQLite veritabanı bağlantısını, tablo oluşturma işlemlerini
ve CRUD operasyonlarını yönetir.

Kullanım:
    from backend.database.db_manager import DatabaseManager
    from backend.core.config import settings
    
    db = DatabaseManager(settings.DATABASE_PATH)
    db.initialize_db()
    
    # Veri ekle
    db.bulk_insert_matches(dataframe)
    
    # Sorgu çalıştır
    results = db.execute_query("SELECT * FROM matches_history LIMIT 10")
    
    # Bağlantıyı kapat
    db.close()

Tablolar:
    - matches_history: Geçmiş maç verileri (Eğitim seti)
    - fixtures: Oynanmamış gelecek maçlar
    - predictions: Model tahminleri
    - wallet_simulation: Backtest finansal kayıtları
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from contextlib import contextmanager
from dataclasses import dataclass, asdict

import pandas as pd
from loguru import logger


# ============================================================================
# VERI MODELLERİ (Data Classes)
# ============================================================================

@dataclass
class MatchHistory:
    """Geçmiş maç verisi modeli"""
    id: Optional[int] = None
    date: str = ""
    home_team: str = ""
    away_team: str = ""
    division: Optional[str] = None  # Lig kodu (E0, T1, D1 vb.)
    tier: int = 1  # Lig seviyesi (1 = Ana Lig, 2 = Alt Lig)
    referee: Optional[str] = None  # Hakem adı
    fthg: int = 0  # Full Time Home Goals
    ftag: int = 0  # Full Time Away Goals
    result: str = ""  # H (Home), D (Draw), A (Away)
    stats: Optional[str] = None  # JSON formatında istatistikler (40 parametre)
    home_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    away_odds: Optional[float] = None
    season: Optional[str] = None


@dataclass
class Fixture:
    """Oynanmamış maç modeli"""
    id: Optional[int] = None
    date: str = ""
    time: Optional[str] = None
    home_team: str = ""
    away_team: str = ""
    status: str = "pending"  # pending, completed, postponed


@dataclass
class Prediction:
    """Tahmin modeli"""
    id: Optional[int] = None
    match_id: int = 0
    prob_home: float = 0.0
    prob_draw: float = 0.0
    prob_away: float = 0.0
    predicted_home_goals: Optional[int] = None
    predicted_away_goals: Optional[int] = None
    ai_risk_analysis: Optional[str] = None
    is_value: bool = False
    confidence_score: float = 0.0
    bet_result: Optional[str] = None  # win, lose, void
    created_at: Optional[str] = None


@dataclass
class WalletTransaction:
    """Cüzdan işlem modeli"""
    id: Optional[int] = None
    date: str = ""
    match_id: int = 0
    bet_type: str = ""  # MS1, MS0, MS2, ALT25, UST25, vb.
    stake: float = 0.0
    odds: float = 0.0
    pnl: float = 0.0  # Profit/Loss
    balance: float = 0.0


# ============================================================================
# SQL ŞEMALARI
# ============================================================================

SCHEMA_MATCHES_HISTORY = """
CREATE TABLE IF NOT EXISTS matches_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    division TEXT,
    tier INTEGER DEFAULT 1,
    referee TEXT,
    fthg INTEGER NOT NULL,
    ftag INTEGER NOT NULL,
    result TEXT NOT NULL CHECK(result IN ('H', 'D', 'A')),
    stats TEXT,
    home_odds REAL,
    draw_odds REAL,
    away_odds REAL,
    season TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, home_team, away_team, division)
);
"""

SCHEMA_FIXTURES = """
CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    time TEXT,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'completed', 'postponed')),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, home_team, away_team)
);
"""

SCHEMA_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER NOT NULL,
    prob_home REAL NOT NULL,
    prob_draw REAL NOT NULL,
    prob_away REAL NOT NULL,
    predicted_home_goals INTEGER,
    predicted_away_goals INTEGER,
    ai_risk_analysis TEXT,
    is_value INTEGER NOT NULL DEFAULT 0,
    confidence_score REAL NOT NULL DEFAULT 0.0,
    bet_result TEXT CHECK(bet_result IN ('win', 'lose', 'void', NULL)),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES fixtures(id) ON DELETE CASCADE
);
"""

SCHEMA_WALLET_SIMULATION = """
CREATE TABLE IF NOT EXISTS wallet_simulation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    match_id INTEGER NOT NULL,
    bet_type TEXT NOT NULL,
    stake REAL NOT NULL,
    odds REAL NOT NULL,
    pnl REAL NOT NULL,
    balance REAL NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

# İndeksler (Performans optimizasyonu)
INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches_history(date);",
    "CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches_history(home_team);",
    "CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches_history(away_team);",
    "CREATE INDEX IF NOT EXISTS idx_matches_season ON matches_history(season);",
    "CREATE INDEX IF NOT EXISTS idx_matches_division ON matches_history(division);",
    "CREATE INDEX IF NOT EXISTS idx_matches_tier ON matches_history(tier);",
    "CREATE INDEX IF NOT EXISTS idx_matches_referee ON matches_history(referee);",
    "CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures(date);",
    "CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures(status);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id);",
    "CREATE INDEX IF NOT EXISTS idx_predictions_is_value ON predictions(is_value);",
    "CREATE INDEX IF NOT EXISTS idx_wallet_date ON wallet_simulation(date);",
]


# ============================================================================
# VERİTABANI YÖNETİCİSİ SINIFI
# ============================================================================

class DatabaseManager:
    """
    SQLite veritabanı yönetim sınıfı.
    
    Bu sınıf, veritabanı bağlantısını, tablo oluşturma işlemlerini
    ve tüm CRUD operasyonlarını yönetir.
    
    Attributes:
        db_path (Path): Veritabanı dosyasının yolu
        connection (sqlite3.Connection): Aktif veritabanı bağlantısı
        
    Example:
        >>> db = DatabaseManager(Path("data/oracle.db"))
        >>> db.initialize_db()
        >>> db.bulk_insert_matches(df)
        >>> db.close()
    """
    
    def __init__(
        self, 
        db_path: Path,
        timeout: int = 30,
        check_same_thread: bool = False
    ) -> None:
        """
        DatabaseManager sınıfını başlatır.
        
        Args:
            db_path: Veritabanı dosyasının yolu
            timeout: Bağlantı zaman aşımı (saniye)
            check_same_thread: Aynı thread kontrolü (False = multi-thread erişim)
        """
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.check_same_thread = check_same_thread
        self._connection: Optional[sqlite3.Connection] = None
        
        logger.info(f"DatabaseManager başlatıldı: {self.db_path}")
    
    @property
    def connection(self) -> sqlite3.Connection:
        """
        Aktif veritabanı bağlantısını döndürür.
        Bağlantı yoksa yeni bir bağlantı oluşturur.
        
        Returns:
            sqlite3.Connection: Veritabanı bağlantısı
        """
        if self._connection is None:
            self._connection = self._create_connection()
        return self._connection
    
    def _create_connection(self) -> sqlite3.Connection:
        """
        Yeni bir SQLite bağlantısı oluşturur.
        
        Returns:
            sqlite3.Connection: Yeni veritabanı bağlantısı
            
        Raises:
            sqlite3.Error: Bağlantı hatası
        """
        try:
            # Veritabanı dosyasının bulunduğu dizini oluştur
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=self.timeout,
                check_same_thread=self.check_same_thread
            )
            
            # Row factory ayarla (dict benzeri erişim için)
            conn.row_factory = sqlite3.Row
            
            # Foreign key desteğini aktifleştir
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # WAL mode (Write-Ahead Logging) - daha iyi eşzamanlılık
            conn.execute("PRAGMA journal_mode = WAL;")
            
            logger.debug(f"Veritabanı bağlantısı kuruldu: {self.db_path}")
            return conn
            
        except sqlite3.Error as e:
            logger.error(f"Veritabanı bağlantı hatası: {e}")
            raise
    
    @contextmanager
    def get_cursor(self):
        """
        Context manager ile cursor sağlar.
        Otomatik commit ve rollback yönetimi yapar.
        
        Yields:
            sqlite3.Cursor: Veritabanı cursor'u
            
        Example:
            >>> with db.get_cursor() as cursor:
            ...     cursor.execute("SELECT * FROM matches_history")
            ...     rows = cursor.fetchall()
        """
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except sqlite3.Error as e:
            self.connection.rollback()
            logger.error(f"Veritabanı işlem hatası: {e}")
            raise
        finally:
            cursor.close()
    
    def initialize_db(self) -> bool:
        """
        Veritabanı tablolarını ve indeksleri oluşturur.
        
        Returns:
            bool: Başarılı ise True
            
        Raises:
            sqlite3.Error: Tablo oluşturma hatası
        """
        try:
            with self.get_cursor() as cursor:
                # Tabloları oluştur
                logger.info("Veritabanı tabloları oluşturuluyor...")
                
                cursor.execute(SCHEMA_MATCHES_HISTORY)
                logger.debug("matches_history tablosu oluşturuldu")
                
                cursor.execute(SCHEMA_FIXTURES)
                logger.debug("fixtures tablosu oluşturuldu")
                
                cursor.execute(SCHEMA_PREDICTIONS)
                logger.debug("predictions tablosu oluşturuldu")
                
                cursor.execute(SCHEMA_WALLET_SIMULATION)
                logger.debug("wallet_simulation tablosu oluşturuldu")
                
                # İndeksleri oluştur
                logger.info("Veritabanı indeksleri oluşturuluyor...")
                for index_sql in INDEXES:
                    cursor.execute(index_sql)
                
                logger.info("Veritabanı başarıyla başlatıldı")
                return True
                
        except sqlite3.Error as e:
            logger.error(f"Veritabanı başlatma hatası: {e}")
            raise
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        SQL sorgusu çalıştırır ve sonuçları döndürür.
        
        Args:
            query: SQL sorgusu
            params: Sorgu parametreleri (tuple)
            
        Returns:
            List[Dict[str, Any]]: Sorgu sonuçları
            
        Example:
            >>> results = db.execute_query(
            ...     "SELECT * FROM matches_history WHERE home_team = ?",
            ...     ("Arsenal",)
            ... )
        """
        try:
            with self.get_cursor() as cursor:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # SELECT sorgusu ise sonuçları döndür
                if query.strip().upper().startswith("SELECT"):
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                
                return []
                
        except sqlite3.Error as e:
            logger.error(f"Sorgu hatası: {e}\nSorgu: {query}")
            raise
    
    def execute_many(
        self, 
        query: str, 
        params_list: List[Tuple]
    ) -> int:
        """
        Birden fazla kayıt için batch insert yapar.
        
        Args:
            query: INSERT SQL sorgusu
            params_list: Parametre listesi
            
        Returns:
            int: Eklenen kayıt sayısı
        """
        try:
            with self.get_cursor() as cursor:
                cursor.executemany(query, params_list)
                return cursor.rowcount
                
        except sqlite3.Error as e:
            logger.error(f"Batch insert hatası: {e}")
            raise
    
    # ========================================================================
    # MATCHES_HISTORY OPERASYONLARI
    # ========================================================================
    
    def bulk_insert_matches(
        self, 
        df: pd.DataFrame, 
        replace_existing: bool = False
    ) -> int:
        """
        Pandas DataFrame'den matches_history tablosuna toplu veri ekler.
        
        Args:
            df: Maç verilerini içeren DataFrame
                Gerekli sütunlar: date, home_team, away_team, fthg, ftag, result
                Opsiyonel sütunlar: division, tier, referee, stats, oranlar
            replace_existing: Var olan kayıtları güncelle (True) veya atla (False)
            
        Returns:
            int: Eklenen kayıt sayısı
            
        Raises:
            ValueError: Gerekli sütunlar eksikse
            sqlite3.Error: Veritabanı hatası
            
        Example:
            >>> df = pd.read_csv("matches.csv")
            >>> count = db.bulk_insert_matches(df)
            >>> print(f"{count} kayıt eklendi")
        """
        required_columns = ['date', 'home_team', 'away_team', 'fthg', 'ftag', 'result']
        
        # Sütun kontrolü
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Eksik sütunlar: {missing_cols}")
        
        # NaN değerleri temizle
        df = df.copy()
        df = df.dropna(subset=required_columns)
        
        if df.empty:
            logger.warning("Eklenecek veri bulunamadı (DataFrame boş)")
            return 0
        
        # INSERT sorgusu - yeni sütunlar eklendi
        if replace_existing:
            insert_sql = """
                INSERT OR REPLACE INTO matches_history 
                (date, home_team, away_team, division, tier, referee, 
                 fthg, ftag, result, stats, home_odds, draw_odds, away_odds, season)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        else:
            insert_sql = """
                INSERT OR IGNORE INTO matches_history 
                (date, home_team, away_team, division, tier, referee,
                 fthg, ftag, result, stats, home_odds, draw_odds, away_odds, season)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        
        # Veriyi tuple listesine dönüştür
        records = []
        for _, row in df.iterrows():
            # Stats JSON'u oluştur (40 parametre desteği)
            stats_dict = {}
            
            # Temel istatistikler
            stat_columns = ['hs', 'as', 'hst', 'ast', 'hc', 'ac', 'hf', 'af', 'hy', 'ay', 'hr', 'ar']
            for col in stat_columns:
                if col in df.columns and pd.notna(row.get(col)):
                    try:
                        stats_dict[col.upper()] = int(row[col])
                    except (ValueError, TypeError):
                        pass
            
            # İlk yarı skorları
            for col in ['hthg', 'htag', 'htr']:
                if col in df.columns and pd.notna(row.get(col)):
                    if col == 'htr':
                        stats_dict[col.upper()] = str(row[col])
                    else:
                        try:
                            stats_dict[col.upper()] = int(row[col])
                        except (ValueError, TypeError):
                            pass
            
            # Genişletilmiş oran verileri
            odds_cols = [
                'b365h', 'b365d', 'b365a',  # Bet365
                'avgh', 'avgd', 'avga',      # Average
                'maxh', 'maxd', 'maxa',      # Max
                'b365ch', 'b365cd', 'b365ca',  # Closing
                'b365>2.5', 'b365<2.5',      # Over/Under
                'ahh', 'b365ahh', 'b365aha'  # Asian Handicap
            ]
            for col in odds_cols:
                if col in df.columns and pd.notna(row.get(col)):
                    try:
                        stats_dict[col.upper().replace('.', '_').replace('<', 'U').replace('>', 'O')] = float(row[col])
                    except (ValueError, TypeError):
                        pass
            
            stats_json = json.dumps(stats_dict) if stats_dict else None
            
            # Güvenli değer alma fonksiyonları
            def safe_get_float(col_name):
                if col_name not in df.columns:
                    return None
                try:
                    val = row[col_name]
                    if val is None:
                        return None
                    import math
                    if isinstance(val, float) and math.isnan(val):
                        return None
                    return float(val)
                except (ValueError, TypeError, KeyError):
                    return None
            
            def safe_get_str(col_name):
                if col_name not in df.columns:
                    return None
                try:
                    val = row[col_name]
                    if val is None:
                        return None
                    import math
                    if isinstance(val, float) and math.isnan(val):
                        return None
                    return str(val).strip() if val else None
                except (ValueError, TypeError, KeyError):
                    return None
            
            def safe_get_int(col_name, default=None):
                if col_name not in df.columns:
                    return default
                try:
                    val = row[col_name]
                    if val is None:
                        return default
                    import math
                    if isinstance(val, float) and math.isnan(val):
                        return default
                    return int(val)
                except (ValueError, TypeError, KeyError):
                    return default
            
            record = (
                str(row['date']),
                str(row['home_team']),
                str(row['away_team']),
                safe_get_str('division'),
                safe_get_int('tier', 1),
                safe_get_str('referee'),
                int(row['fthg']),
                int(row['ftag']),
                str(row['result']),
                stats_json,
                safe_get_float('home_odds'),
                safe_get_float('draw_odds'),
                safe_get_float('away_odds'),
                safe_get_str('season'),
            )
            records.append(record)
        
        try:
            inserted = self.execute_many(insert_sql, records)
            logger.info(f"{inserted} maç kaydı eklendi (toplam: {len(records)})")
            return inserted
            
        except sqlite3.Error as e:
            logger.error(f"Toplu ekleme hatası: {e}")
            raise
    
    def get_matches_by_team(
        self, 
        team_name: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Belirli bir takımın maçlarını getirir.
        
        Args:
            team_name: Takım adı
            limit: Maksimum kayıt sayısı
            
        Returns:
            List[Dict]: Maç kayıtları
        """
        query = """
            SELECT * FROM matches_history
            WHERE home_team = ? OR away_team = ?
            ORDER BY date DESC
            LIMIT ?
        """
        return self.execute_query(query, (team_name, team_name, limit))
    
    def get_matches_by_season(self, season: str) -> List[Dict[str, Any]]:
        """
        Belirli bir sezonun tüm maçlarını getirir.
        
        Args:
            season: Sezon (örn: "2023-2024")
            
        Returns:
            List[Dict]: Maç kayıtları
        """
        query = """
            SELECT * FROM matches_history
            WHERE season = ?
            ORDER BY date ASC
        """
        return self.execute_query(query, (season,))
    
    def get_matches_dataframe(
        self, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Maç verilerini Pandas DataFrame olarak döndürür.
        
        Args:
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Maç verileri
        """
        if start_date and end_date:
            query = """
                SELECT * FROM matches_history
                WHERE date BETWEEN ? AND ?
                ORDER BY date ASC
            """
            params = (start_date, end_date)
        elif start_date:
            query = """
                SELECT * FROM matches_history
                WHERE date >= ?
                ORDER BY date ASC
            """
            params = (start_date,)
        elif end_date:
            query = """
                SELECT * FROM matches_history
                WHERE date <= ?
                ORDER BY date ASC
            """
            params = (end_date,)
        else:
            query = "SELECT * FROM matches_history ORDER BY date ASC"
            params = None
        
        results = self.execute_query(query, params) if params else self.execute_query(query)
        return pd.DataFrame(results)
    
    def get_match_count(self) -> int:
        """Toplam maç sayısını döndürür."""
        result = self.execute_query("SELECT COUNT(*) as count FROM matches_history")
        return result[0]['count'] if result else 0
    
    # ========================================================================
    # FIXTURES OPERASYONLARI
    # ========================================================================
    
    def insert_fixture(self, fixture: Fixture) -> int:
        """
        Yeni bir fikstür ekler.
        
        Args:
            fixture: Fixture nesnesi
            
        Returns:
            int: Eklenen kaydın ID'si
        """
        query = """
            INSERT OR IGNORE INTO fixtures (date, time, home_team, away_team, status)
            VALUES (?, ?, ?, ?, ?)
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (
                fixture.date,
                fixture.time,
                fixture.home_team,
                fixture.away_team,
                fixture.status
            ))
            return cursor.lastrowid
    
    def get_pending_fixtures(self) -> List[Dict[str, Any]]:
        """Oynanmamış maçları getirir."""
        query = """
            SELECT * FROM fixtures
            WHERE status = 'pending'
            ORDER BY date ASC
        """
        return self.execute_query(query)
    
    def update_fixture_status(self, fixture_id: int, status: str) -> bool:
        """
        Fikstür durumunu günceller.
        
        Args:
            fixture_id: Fikstür ID'si
            status: Yeni durum (pending, completed, postponed)
            
        Returns:
            bool: Başarılı ise True
        """
        query = """
            UPDATE fixtures
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        self.execute_query(query, (status, fixture_id))
        return True
    
    # ========================================================================
    # PREDICTIONS OPERASYONLARI
    # ========================================================================
    
    def insert_prediction(self, prediction: Prediction) -> int:
        """
        Yeni bir tahmin ekler.
        
        Args:
            prediction: Prediction nesnesi
            
        Returns:
            int: Eklenen kaydın ID'si
        """
        query = """
            INSERT INTO predictions 
            (match_id, prob_home, prob_draw, prob_away, predicted_home_goals,
             predicted_away_goals, ai_risk_analysis, is_value, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (
                prediction.match_id,
                prediction.prob_home,
                prediction.prob_draw,
                prediction.prob_away,
                prediction.predicted_home_goals,
                prediction.predicted_away_goals,
                prediction.ai_risk_analysis,
                1 if prediction.is_value else 0,
                prediction.confidence_score
            ))
            return cursor.lastrowid
    
    def get_value_bets(self) -> List[Dict[str, Any]]:
        """Value bet olarak işaretlenmiş tahminleri getirir."""
        query = """
            SELECT p.*, f.home_team, f.away_team, f.date
            FROM predictions p
            JOIN fixtures f ON p.match_id = f.id
            WHERE p.is_value = 1
            ORDER BY p.confidence_score DESC
        """
        return self.execute_query(query)
    
    # ========================================================================
    # WALLET OPERASYONLARI
    # ========================================================================
    
    def insert_wallet_transaction(self, transaction: WalletTransaction) -> int:
        """
        Cüzdan işlemi ekler.
        
        Args:
            transaction: WalletTransaction nesnesi
            
        Returns:
            int: Eklenen kaydın ID'si
        """
        query = """
            INSERT INTO wallet_simulation
            (date, match_id, bet_type, stake, odds, pnl, balance)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (
                transaction.date,
                transaction.match_id,
                transaction.bet_type,
                transaction.stake,
                transaction.odds,
                transaction.pnl,
                transaction.balance
            ))
            return cursor.lastrowid
    
    def get_wallet_balance(self) -> float:
        """Son cüzdan bakiyesini döndürür."""
        query = """
            SELECT balance FROM wallet_simulation
            ORDER BY id DESC
            LIMIT 1
        """
        result = self.execute_query(query)
        return result[0]['balance'] if result else 0.0
    
    def get_wallet_summary(self) -> Dict[str, Any]:
        """
        Cüzdan özet istatistiklerini döndürür.
        
        Returns:
            Dict: ROI, toplam bahis, toplam kar/zarar vb.
        """
        query = """
            SELECT 
                COUNT(*) as total_bets,
                SUM(stake) as total_staked,
                SUM(pnl) as total_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
                MAX(balance) as peak_balance,
                MIN(balance) as lowest_balance
            FROM wallet_simulation
        """
        result = self.execute_query(query)
        
        if result and result[0]['total_bets']:
            data = result[0]
            roi = (data['total_pnl'] / data['total_staked'] * 100) if data['total_staked'] else 0
            hit_rate = (data['wins'] / data['total_bets'] * 100) if data['total_bets'] else 0
            
            return {
                'total_bets': data['total_bets'],
                'total_staked': data['total_staked'],
                'total_pnl': data['total_pnl'],
                'roi': round(roi, 2),
                'wins': data['wins'],
                'losses': data['losses'],
                'hit_rate': round(hit_rate, 2),
                'peak_balance': data['peak_balance'],
                'lowest_balance': data['lowest_balance'],
                'max_drawdown': round(((data['peak_balance'] - data['lowest_balance']) / data['peak_balance'] * 100), 2) if data['peak_balance'] else 0
            }
        
        return {
            'total_bets': 0,
            'total_staked': 0,
            'total_pnl': 0,
            'roi': 0,
            'wins': 0,
            'losses': 0,
            'hit_rate': 0,
            'peak_balance': 0,
            'lowest_balance': 0,
            'max_drawdown': 0
        }
    
    # ========================================================================
    # YARDIMCI METODLAR
    # ========================================================================
    
    def table_exists(self, table_name: str) -> bool:
        """Tablonun var olup olmadığını kontrol eder."""
        query = """
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name=?
        """
        result = self.execute_query(query, (table_name,))
        return len(result) > 0
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Tablo şema bilgisini döndürür."""
        query = f"PRAGMA table_info({table_name})"
        return self.execute_query(query)
    
    def vacuum(self) -> None:
        """Veritabanını optimize eder."""
        self.connection.execute("VACUUM")
        logger.info("Veritabanı optimize edildi (VACUUM)")
    
    def backup(self, backup_path: Path) -> bool:
        """
        Veritabanı yedeği alır.
        
        Args:
            backup_path: Yedek dosyasının yolu
            
        Returns:
            bool: Başarılı ise True
        """
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            backup_conn = sqlite3.connect(str(backup_path))
            self.connection.backup(backup_conn)
            backup_conn.close()
            
            logger.info(f"Veritabanı yedeği alındı: {backup_path}")
            return True
            
        except sqlite3.Error as e:
            logger.error(f"Yedekleme hatası: {e}")
            return False
    
    def close(self) -> None:
        """Veritabanı bağlantısını kapatır."""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Veritabanı bağlantısı kapatıldı")
    
    def __enter__(self) -> "DatabaseManager":
        """Context manager giriş"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager çıkış"""
        self.close()
    
    def __del__(self) -> None:
        """Destructor - bağlantıyı temizle"""
        self.close()


# ============================================================================
# YARDIMCI FONKSİYONLAR
# ============================================================================

def get_database() -> DatabaseManager:
    """
    Varsayılan ayarlarla DatabaseManager instance'ı döndürür.
    
    Returns:
        DatabaseManager: Veritabanı yöneticisi
        
    Example:
        >>> from backend.database.db_manager import get_database
        >>> db = get_database()
        >>> db.initialize_db()
    """
    from backend.core.config import settings
    
    return DatabaseManager(
        db_path=settings.DATABASE_PATH,
        timeout=settings.DB_TIMEOUT,
        check_same_thread=settings.DB_CHECK_SAME_THREAD
    )
