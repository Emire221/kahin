"""
The Oracle - Football-Data.org API Veri Çekici

Bu script, football-data.org API'sinden Premier Lig maç verilerini çeker
ve veritabanına yükler.

API Dokümantasyonu: https://www.football-data.org/documentation/api

Kullanım:
    python -m backend.data.api_fetcher fetch --seasons 2023,2022,2021
    python -m backend.data.api_fetcher fetch --all
"""

import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
from loguru import logger

from backend.core.config import settings
from backend.database.db_manager import get_database


class FootballDataAPI:
    """
    Football-Data.org API istemcisi.
    
    Premier Lig maç verilerini API üzerinden çeker.
    """
    
    BASE_URL = "https://api.football-data.org/v4"
    COMPETITION_CODE = "PL"  # Premier League
    
    # Rate limiting: Free tier 10 requests/minute
    REQUEST_DELAY = 6.5  # saniye
    
    def __init__(self, api_key: str) -> None:
        """
        API istemcisini başlatır.
        
        Args:
            api_key: Football-data.org API anahtarı
        """
        self.api_key = api_key
        self.headers = {
            "X-Auth-Token": api_key
        }
        self._last_request_time = 0
        
        logger.info("Football-Data API istemcisi başlatıldı")
    
    def _rate_limit(self) -> None:
        """Rate limiting uygular"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.REQUEST_DELAY:
            wait_time = self.REQUEST_DELAY - elapsed
            logger.debug(f"Rate limit: {wait_time:.1f}s bekleniyor...")
            time.sleep(wait_time)
        self._last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        API isteği yapar.
        
        Args:
            endpoint: API endpoint'i
            params: Query parametreleri
            
        Returns:
            Dict: API yanıtı
        """
        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning("Rate limit aşıldı, 60 saniye bekleniyor...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            elif response.status_code == 403:
                logger.error("API anahtarı geçersiz veya yetki yok!")
                raise
            else:
                logger.error(f"HTTP Hatası: {e}")
                raise
        except requests.exceptions.RequestException as e:
            logger.error(f"İstek hatası: {e}")
            raise
    
    def get_available_seasons(self) -> List[int]:
        """
        Mevcut sezonları döndürür.
        
        Returns:
            List[int]: Sezon yılları listesi
        """
        data = self._make_request(f"/competitions/{self.COMPETITION_CODE}")
        
        seasons = []
        if 'seasons' in data:
            for season in data['seasons']:
                year = int(season['startDate'][:4])
                seasons.append(year)
        
        return sorted(seasons, reverse=True)
    
    def get_matches(self, season: int) -> List[Dict]:
        """
        Belirli bir sezonun maçlarını çeker.
        
        Args:
            season: Sezon başlangıç yılı (örn: 2023 = 2023-2024 sezonu)
            
        Returns:
            List[Dict]: Maç listesi
        """
        logger.info(f"Sezon {season}-{season+1} maçları çekiliyor...")
        
        data = self._make_request(
            f"/competitions/{self.COMPETITION_CODE}/matches",
            params={"season": season}
        )
        
        matches = data.get('matches', [])
        logger.info(f"  {len(matches)} maç bulundu")
        
        return matches
    
    def get_standings(self, season: int) -> Dict:
        """Puan durumunu çeker"""
        data = self._make_request(
            f"/competitions/{self.COMPETITION_CODE}/standings",
            params={"season": season}
        )
        return data
    
    def convert_to_dataframe(self, matches: List[Dict]) -> pd.DataFrame:
        """
        API yanıtını DataFrame'e dönüştürür.
        
        Args:
            matches: API'den gelen maç listesi
            
        Returns:
            pd.DataFrame: Standart formatta maç verileri
        """
        records = []
        
        for match in matches:
            # Sadece tamamlanmış maçları al
            if match.get('status') != 'FINISHED':
                continue
            
            score = match.get('score', {})
            full_time = score.get('fullTime', {})
            
            # Skor bilgisi yoksa atla
            if full_time.get('home') is None or full_time.get('away') is None:
                continue
            
            home_goals = full_time['home']
            away_goals = full_time['away']
            
            # Sonucu hesapla
            if home_goals > away_goals:
                result = 'H'
            elif home_goals < away_goals:
                result = 'A'
            else:
                result = 'D'
            
            # Tarihi parse et
            utc_date = match.get('utcDate', '')
            if utc_date:
                date = utc_date[:10]  # YYYY-MM-DD
            else:
                continue
            
            # Sezon bilgisi
            season_data = match.get('season', {})
            season_start = season_data.get('startDate', '')[:4]
            season_end = season_data.get('endDate', '')[:4]
            season_str = f"{season_start}-{season_end}" if season_start and season_end else None
            
            record = {
                'date': date,
                'home_team': match.get('homeTeam', {}).get('shortName', match.get('homeTeam', {}).get('name', '')),
                'away_team': match.get('awayTeam', {}).get('shortName', match.get('awayTeam', {}).get('name', '')),
                'fthg': home_goals,
                'ftag': away_goals,
                'result': result,
                'season': season_str,
                'matchday': match.get('matchday'),
                # Odds bilgisi API'de yok, None olarak bırak
                'home_odds': None,
                'draw_odds': None,
                'away_odds': None,
            }
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        if not df.empty:
            df = df.sort_values('date')
        
        return df
    
    def fetch_and_save(
        self, 
        seasons: List[int],
        save_to_db: bool = True,
        save_to_csv: bool = True
    ) -> int:
        """
        Belirtilen sezonların verilerini çeker ve kaydeder.
        
        Args:
            seasons: Sezon yılları listesi
            save_to_db: Veritabanına kaydet
            save_to_csv: CSV olarak da kaydet
            
        Returns:
            int: Toplam kaydedilen maç sayısı
        """
        all_matches = []
        
        for season in seasons:
            try:
                matches = self.get_matches(season)
                df = self.convert_to_dataframe(matches)
                
                if not df.empty:
                    all_matches.append(df)
                    
                    # CSV olarak kaydet
                    if save_to_csv:
                        csv_path = settings.RAW_CSV_DIR / f"PL_{season}_{season+1}.csv"
                        df.to_csv(csv_path, index=False)
                        logger.info(f"  CSV kaydedildi: {csv_path.name}")
                
            except Exception as e:
                logger.error(f"Sezon {season} çekilemedi: {e}")
                continue
        
        if not all_matches:
            logger.warning("Hiç maç verisi çekilemedi!")
            return 0
        
        # Tüm verileri birleştir
        combined_df = pd.concat(all_matches, ignore_index=True)
        
        # Veritabanına kaydet
        if save_to_db:
            from backend.data.data_loader import DataLoader
            
            with DataLoader() as loader:
                # Takım isimlerini standardize et
                combined_df = loader.standardize_team_names(combined_df)
                count = loader.load_to_database(combined_df, replace_existing=True)
                logger.info(f"Toplam {count} maç veritabanına kaydedildi")
                return count
        
        return len(combined_df)


def fetch_premier_league_data(
    api_key: str,
    seasons: Optional[List[int]] = None,
    years_back: int = 10
) -> int:
    """
    Premier Lig verilerini çeker ve kaydeder.
    
    Args:
        api_key: Football-data.org API anahtarı
        seasons: Çekilecek sezonlar (None ise son N yıl)
        years_back: Kaç yıl geriye git
        
    Returns:
        int: Kaydedilen maç sayısı
    """
    api = FootballDataAPI(api_key)
    
    if seasons is None:
        # Mevcut sezonları al ve son N yılı seç
        available = api.get_available_seasons()
        seasons = available[:years_back]
    
    logger.info(f"Çekilecek sezonlar: {seasons}")
    
    return api.fetch_and_save(seasons)


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Football-Data.org API Veri Çekici")
    
    subparsers = parser.add_subparsers(dest='command')
    
    # fetch komutu
    fetch_parser = subparsers.add_parser('fetch', help='Veri çek')
    fetch_parser.add_argument('--api-key', type=str, help='API anahtarı')
    fetch_parser.add_argument('--seasons', type=str, help='Sezonlar (virgülle ayrılmış)')
    fetch_parser.add_argument('--years', type=int, default=5, help='Kaç yıl geriye git')
    
    args = parser.parse_args()
    
    if args.command == 'fetch':
        api_key = args.api_key or input("API Anahtarı: ")
        
        seasons = None
        if args.seasons:
            seasons = [int(s.strip()) for s in args.seasons.split(',')]
        
        count = fetch_premier_league_data(api_key, seasons, args.years)
        print(f"\n✅ {count} maç başarıyla yüklendi!")


if __name__ == "__main__":
    main()
