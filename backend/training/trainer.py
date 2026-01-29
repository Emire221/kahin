"""
The Oracle - Model Eğitim Modülü

Bu modül, Dixon-Coles ve XGBoost modellerini eğitmek,
kaydetmek ve yönetmek için kullanılır.

Temel Özellikler:
    - Tek komutla tüm modelleri eğitme
    - Model versiyonlama
    - Eğitim raporları
    - Ensemble (birleşik) tahmin

Kullanım:
    from backend.training.trainer import ModelTrainer
    
    trainer = ModelTrainer()
    
    # Tüm modelleri eğit
    trainer.train_all()
    
    # Belirli bir modeli eğit
    trainer.train_dixon_coles()
    trainer.train_xgboost()
    
    # Ensemble tahmin
    probs = trainer.predict_ensemble("Arsenal", "Chelsea")

CLI Kullanımı:
    python -m backend.training.trainer train
    python -m backend.training.trainer predict "Arsenal" "Chelsea"
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
from loguru import logger

from backend.core.config import settings
from backend.database.db_manager import DatabaseManager, get_database
from backend.models.dixon_coles import DixonColesModel
from backend.models.xgboost_model import XGBoostPredictor


class ModelTrainer:
    """
    Model Eğitim ve Yönetim Sınıfı
    
    Bu sınıf, tüm ML modellerinin eğitimini, kaydedilmesini
    ve tahmin işlemlerini merkezi olarak yönetir.
    
    Attributes:
        dixon_coles (DixonColesModel): Skor tahmin modeli
        xgboost (XGBoostPredictor): Sonuç tahmin modeli
        
    Example:
        >>> trainer = ModelTrainer()
        >>> trainer.train_all()
        >>> result = trainer.predict_ensemble("Arsenal", "Chelsea")
    """
    
    # Model dosya isimleri
    DIXON_COLES_FILENAME = "dixon_coles.pkl"
    XGBOOST_FILENAME = "xgboost.pkl"
    TRAINING_REPORT_FILENAME = "training_report.json"
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        models_dir: Optional[Path] = None,
        dixon_coles_weight: float = 0.4,
        xgboost_weight: float = 0.6
    ) -> None:
        """
        ModelTrainer sınıfını başlatır.
        
        Args:
            db_manager: Veritabanı yöneticisi
            models_dir: Model kayıt dizini
            dixon_coles_weight: Ensemble'da Dixon-Coles ağırlığı
            xgboost_weight: Ensemble'da XGBoost ağırlığı
        """
        self.db_manager = db_manager
        self._owns_db = db_manager is None
        
        self.models_dir = Path(models_dir) if models_dir else settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensemble ağırlıkları (toplam 1 olmalı)
        total_weight = dixon_coles_weight + xgboost_weight
        self.dixon_coles_weight = dixon_coles_weight / total_weight
        self.xgboost_weight = xgboost_weight / total_weight
        
        # Modeller
        self.dixon_coles: Optional[DixonColesModel] = None
        self.xgboost: Optional[XGBoostPredictor] = None
        
        # Eğitim bilgileri
        self._training_report: Dict[str, Any] = {}
        
        logger.info(f"ModelTrainer başlatıldı. Model dizini: {self.models_dir}")
    
    def _get_db(self) -> DatabaseManager:
        """Veritabanı yöneticisini döndürür"""
        if self.db_manager is None:
            self.db_manager = get_database()
            self._owns_db = True
        return self.db_manager
    
    def _get_training_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Eğitim verisini veritabanından çeker.
        
        Args:
            start_date: Başlangıç tarihi (YYYY-MM-DD)
            end_date: Bitiş tarihi (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: Maç verileri
        """
        db = self._get_db()
        df = db.get_matches_dataframe(start_date, end_date)
        
        if df.empty:
            raise ValueError("Veritabanında eğitim verisi bulunamadı!")
        
        logger.info(f"Eğitim verisi yüklendi: {len(df)} maç")
        
        return df
    
    def train_dixon_coles(
        self,
        df: Optional[pd.DataFrame] = None,
        save: bool = True
    ) -> DixonColesModel:
        """
        Dixon-Coles modelini eğitir.
        
        Args:
            df: Eğitim verisi (None ise veritabanından çeker)
            save: Modeli kaydet
            
        Returns:
            DixonColesModel: Eğitilmiş model
        """
        logger.info("Dixon-Coles modeli eğitiliyor...")
        
        if df is None:
            df = self._get_training_data()
        
        # Model oluştur ve eğit
        self.dixon_coles = DixonColesModel(
            max_goals=10,
            time_decay=0.0018,
            use_weights=True
        )
        
        start_time = datetime.now()
        self.dixon_coles.fit(df)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Rapor
        self._training_report['dixon_coles'] = {
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': round(training_time, 2),
            'num_matches': len(df),
            'num_teams': len(self.dixon_coles._teams),
            'home_advantage': round(self.dixon_coles.params.home_advantage, 4),
            'rho': round(self.dixon_coles.params.rho, 4)
        }
        
        if save:
            path = self.models_dir / self.DIXON_COLES_FILENAME
            self.dixon_coles.save(path)
        
        logger.info(f"Dixon-Coles eğitimi tamamlandı ({training_time:.1f}s)")
        
        return self.dixon_coles
    
    def train_xgboost(
        self,
        df: Optional[pd.DataFrame] = None,
        save: bool = True
    ) -> XGBoostPredictor:
        """
        XGBoost modelini eğitir.
        
        Args:
            df: Eğitim verisi
            save: Modeli kaydet
            
        Returns:
            XGBoostPredictor: Eğitilmiş model
        """
        logger.info("XGBoost modeli eğitiliyor...")
        
        if df is None:
            df = self._get_training_data()
        
        # Model oluştur ve eğit
        self.xgboost = XGBoostPredictor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            use_calibration=True
        )
        
        start_time = datetime.now()
        self.xgboost.fit(df)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Rapor
        self._training_report['xgboost'] = {
            'training_date': datetime.now().isoformat(),
            'training_time_seconds': round(training_time, 2),
            'num_matches': len(df),
            'num_teams': len(self.xgboost._team_stats),
            'feature_count': len(self.xgboost.feature_names)
        }
        
        if save:
            path = self.models_dir / self.XGBOOST_FILENAME
            self.xgboost.save(path)
        
        logger.info(f"XGBoost eğitimi tamamlandı ({training_time:.1f}s)")
        
        return self.xgboost
    
    def train_all(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Tüm modelleri eğitir.
        
        Args:
            start_date: Eğitim verisi başlangıç tarihi
            end_date: Eğitim verisi bitiş tarihi
            save: Modelleri kaydet
            
        Returns:
            Dict: Eğitim raporu
        """
        logger.info("=" * 50)
        logger.info("TÜM MODELLER EĞİTİLİYOR")
        logger.info("=" * 50)
        
        # Veriyi bir kere çek
        df = self._get_training_data(start_date, end_date)
        
        # Modelleri eğit
        self.train_dixon_coles(df, save)
        self.train_xgboost(df, save)
        
        # Raporu kaydet
        self._training_report['summary'] = {
            'training_date': datetime.now().isoformat(),
            'total_matches': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            }
        }
        
        if save:
            report_path = self.models_dir / self.TRAINING_REPORT_FILENAME
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(self._training_report, f, indent=2, ensure_ascii=False)
            logger.info(f"Eğitim raporu kaydedildi: {report_path}")
        
        logger.info("=" * 50)
        logger.info("TÜM MODELLER EĞİTİLDİ")
        logger.info("=" * 50)
        
        return self._training_report
    
    def train_by_league(
        self,
        divisions: Optional[List[str]] = None,
        tier1_only: bool = True,
        save: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Lig bazlı model eğitimi yapar.
        
        Her lig için ayrı XGBoost modeli eğitir ve kaydeder.
        Tier 2 verileri, Tier 1 liglerdeki takımların geçmişi olarak kullanılır.
        
        Args:
            divisions: Eğitilecek lig kodları listesi (None ise tüm Tier 1 ligler)
            tier1_only: Sadece Tier 1 ligleri eğit
            save: Modelleri kaydet
            
        Returns:
            Dict: Lig bazlı eğitim raporları
            
        Example:
            >>> trainer.train_by_league(divisions=['E0', 'T1'])
            >>> # veya
            >>> trainer.train_by_league()  # Tüm Tier 1 ligler
        """
        logger.info("=" * 50)
        logger.info("LİG BAZLI EĞİTİM BAŞLIYOR")
        logger.info("=" * 50)
        
        # Tier sabitleri
        TIER_1_LEAGUES = {'E0', 'D1', 'I1', 'SP1', 'F1', 'T1', 'N1', 'B1', 'P1'}
        TIER_2_LEAGUES = {'E1', 'D2', 'I2', 'SP2', 'F2'}
        
        # Veritabanından tüm veriyi çek
        db = self._get_db()
        full_df = db.get_matches_dataframe()
        
        if full_df.empty:
            raise ValueError("Veritabanında eğitim verisi bulunamadı!")
        
        # Division kontrolü
        if 'division' not in full_df.columns:
            logger.warning("Division sütunu bulunamadı, standart eğitim yapılacak")
            return {'default': self.train_all(save=save)}
        
        # Mevcut ligleri bul
        available_divisions = set(full_df['division'].dropna().unique())
        logger.info(f"Mevcut ligler: {available_divisions}")
        
        # Eğitilecek ligleri belirle
        if divisions:
            target_leagues = set(divisions) & available_divisions
        elif tier1_only:
            target_leagues = TIER_1_LEAGUES & available_divisions
        else:
            target_leagues = available_divisions
        
        if not target_leagues:
            raise ValueError("Eğitilecek lig bulunamadı!")
        
        logger.info(f"Eğitilecek ligler ({len(target_leagues)}): {sorted(target_leagues)}")
        
        # Tier 2 verisini hazırla (her Tier 1 eğitiminde context olarak kullanılacak)
        tier2_df = full_df[full_df['division'].isin(TIER_2_LEAGUES)]
        logger.info(f"Tier 2 verisi: {len(tier2_df)} maç")
        
        league_reports = {}
        
        for division in sorted(target_leagues):
            logger.info(f"\n--- {division} Ligi Eğitimi ---")
            
            # Bu lig verisini çek
            league_df = full_df[full_df['division'] == division]
            
            if league_df.empty or len(league_df) < 100:
                logger.warning(f"{division} ligi için yeterli veri yok ({len(league_df)} maç), atlanıyor")
                continue
            
            # Tier 2 verisi ile birleştir (aynı ülke/federasyon)
            # Örn: E0 için E1 verisini de ekle
            related_tier2 = f"{division[0]}2" if len(division) == 2 and division[1] == '0' else None
            
            if related_tier2 and related_tier2 in TIER_2_LEAGUES:
                related_df = tier2_df[tier2_df['division'] == related_tier2]
                combined_df = pd.concat([related_df, league_df], ignore_index=True)
                logger.info(f"{division} + {related_tier2} birleştirildi: {len(combined_df)} maç")
            else:
                combined_df = league_df
            
            # Tarihe göre sırala
            combined_df = combined_df.sort_values('date')
            
            # XGBoost modeli eğit
            try:
                xgb_model = XGBoostPredictor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    use_calibration=True
                )
                
                start_time = datetime.now()
                xgb_model.fit(combined_df)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Kaydet
                if save:
                    model_filename = f"xgboost_{division}.pkl"
                    model_path = self.models_dir / model_filename
                    xgb_model.save(model_path)
                    logger.info(f"Model kaydedildi: {model_path}")
                
                # Rapor
                league_reports[division] = {
                    'training_date': datetime.now().isoformat(),
                    'training_time_seconds': round(training_time, 2),
                    'num_matches': len(combined_df),
                    'num_teams': len(xgb_model._team_stats),
                    'feature_count': len(xgb_model.feature_names),
                    'tier2_matches': len(combined_df) - len(league_df),
                }
                
                logger.info(f"{division} eğitimi tamamlandı ({training_time:.1f}s)")
                
            except Exception as e:
                logger.error(f"{division} eğitimi başarısız: {e}")
                league_reports[division] = {'error': str(e)}
        
        # Genel raporu kaydet
        if save:
            report_path = self.models_dir / "multi_league_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(league_reports, f, indent=2, ensure_ascii=False)
            logger.info(f"Çoklu lig raporu kaydedildi: {report_path}")
        
        logger.info("=" * 50)
        logger.info(f"LİG BAZLI EĞİTİM TAMAMLANDI ({len(league_reports)} lig)")
        logger.info("=" * 50)
        
        return league_reports
    
    def load_league_model(self, division: str) -> Optional[XGBoostPredictor]:
        """
        Belirli bir lig için eğitilmiş modeli yükler.
        
        Args:
            division: Lig kodu (E0, T1, vb.)
            
        Returns:
            XGBoostPredictor veya None
        """
        model_path = self.models_dir / f"xgboost_{division}.pkl"
        
        if model_path.exists():
            model = XGBoostPredictor.load(model_path)
            logger.info(f"{division} modeli yüklendi")
            return model
        else:
            logger.warning(f"{division} modeli bulunamadı: {model_path}")
            return None
    
    def load_models(self) -> None:
        """Kaydedilmiş modelleri yükler."""
        
        # Dixon-Coles
        dc_path = self.models_dir / self.DIXON_COLES_FILENAME
        if dc_path.exists():
            self.dixon_coles = DixonColesModel.load(dc_path)
            logger.info("Dixon-Coles modeli yüklendi")
        else:
            logger.warning(f"Dixon-Coles modeli bulunamadı: {dc_path}")
        
        # XGBoost
        xgb_path = self.models_dir / self.XGBOOST_FILENAME
        if xgb_path.exists():
            self.xgboost = XGBoostPredictor.load(xgb_path)
            logger.info("XGBoost modeli yüklendi")
        else:
            logger.warning(f"XGBoost modeli bulunamadı: {xgb_path}")
    
    def predict_dixon_coles(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, Any]:
        """
        Dixon-Coles modeli ile tahmin yapar.
        
        Returns:
            Dict: Skor ve sonuç olasılıkları
        """
        if self.dixon_coles is None or not self.dixon_coles.is_fitted:
            raise ValueError("Dixon-Coles modeli hazır değil!")
        
        result = self.dixon_coles.predict_match_result(home_team, away_team)
        expected = self.dixon_coles.predict_expected_goals(home_team, away_team)
        over_under = self.dixon_coles.predict_over_under(home_team, away_team)
        btts = self.dixon_coles.predict_btts(home_team, away_team)
        correct_scores = self.dixon_coles.predict_correct_score(home_team, away_team, top_n=3)
        
        return {
            'model': 'Dixon-Coles',
            'home_team': home_team,
            'away_team': away_team,
            'match_result': result,
            'expected_goals': expected,
            'over_under_2.5': over_under,
            'btts': btts,
            'most_likely_scores': correct_scores
        }
    
    def predict_xgboost(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, Any]:
        """
        XGBoost modeli ile tahmin yapar.
        
        Returns:
            Dict: Sonuç olasılıkları
        """
        if self.xgboost is None or not self.xgboost.is_fitted:
            raise ValueError("XGBoost modeli hazır değil!")
        
        proba = self.xgboost.predict_proba(home_team, away_team)
        prediction = self.xgboost.predict(home_team, away_team)
        
        return {
            'model': 'XGBoost',
            'home_team': home_team,
            'away_team': away_team,
            'match_result': proba,
            'prediction': prediction
        }
    
    def predict_ensemble(
        self,
        home_team: str,
        away_team: str
    ) -> Dict[str, Any]:
        """
        Her iki modelin ağırlıklı ortalamasıyla tahmin yapar.
        
        Bu, dökümantasyonda belirtilen "Ensemble (Birleştirme)"
        yaklaşımını implement eder.
        
        Args:
            home_team: Ev sahibi takım
            away_team: Deplasman takım
            
        Returns:
            Dict: Birleşik tahmin sonuçları
        """
        if self.dixon_coles is None or self.xgboost is None:
            self.load_models()
        
        if self.dixon_coles is None or self.xgboost is None:
            raise ValueError("Modeller yüklenemedi!")
        
        # Bireysel tahminler
        dc_result = self.dixon_coles.predict_match_result(home_team, away_team)
        xgb_result = self.xgboost.predict_proba(home_team, away_team)
        
        # Ensemble (ağırlıklı ortalama)
        ensemble_probs = {
            'home_win': round(
                dc_result['home_win'] * self.dixon_coles_weight +
                xgb_result['home_win'] * self.xgboost_weight,
                4
            ),
            'draw': round(
                dc_result['draw'] * self.dixon_coles_weight +
                xgb_result['draw'] * self.xgboost_weight,
                4
            ),
            'away_win': round(
                dc_result['away_win'] * self.dixon_coles_weight +
                xgb_result['away_win'] * self.xgboost_weight,
                4
            )
        }
        
        # En olası sonuç
        max_prob = max(ensemble_probs.values())
        if ensemble_probs['home_win'] == max_prob:
            prediction = 'H'
            prediction_text = f"{home_team} Kazanır"
        elif ensemble_probs['away_win'] == max_prob:
            prediction = 'A'
            prediction_text = f"{away_team} Kazanır"
        else:
            prediction = 'D'
            prediction_text = "Beraberlik"
        
        # Dixon-Coles'tan ek bilgiler
        expected_goals = self.dixon_coles.predict_expected_goals(home_team, away_team)
        over_under = self.dixon_coles.predict_over_under(home_team, away_team)
        btts = self.dixon_coles.predict_btts(home_team, away_team)
        correct_scores = self.dixon_coles.predict_correct_score(home_team, away_team, top_n=3)
        
        # Güven skoru (en yüksek olasılık)
        confidence = max_prob
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'ensemble_probabilities': ensemble_probs,
            'prediction': prediction,
            'prediction_text': prediction_text,
            'confidence': round(confidence, 4),
            'expected_goals': expected_goals,
            'over_under_2.5': over_under,
            'btts': btts,
            'most_likely_scores': correct_scores,
            'model_weights': {
                'dixon_coles': self.dixon_coles_weight,
                'xgboost': self.xgboost_weight
            },
            'individual_predictions': {
                'dixon_coles': dc_result,
                'xgboost': xgb_result
            }
        }
    
    def get_team_analysis(self, team: str) -> Dict[str, Any]:
        """
        Tek bir takımın detaylı analizini döndürür.
        
        Args:
            team: Takım adı
            
        Returns:
            Dict: Takım analizi
        """
        if self.dixon_coles is None or self.xgboost is None:
            self.load_models()
        
        analysis = {
            'team': team,
            'timestamp': datetime.now().isoformat()
        }
        
        # Dixon-Coles parametreleri
        if self.dixon_coles and team in self.dixon_coles.params.teams:
            tp = self.dixon_coles.params.teams[team]
            analysis['dixon_coles'] = {
                'attack_strength': round(tp.attack, 3),
                'defense_strength': round(tp.defense, 3),
                'overall_rating': round(tp.attack / tp.defense, 3)
            }
        
        # XGBoost istatistikleri
        if self.xgboost and team in self.xgboost._team_stats:
            stats = self.xgboost._team_stats[team]
            analysis['xgboost'] = {
                'elo_rating': round(self.xgboost._elo.get_rating(team), 1),
                'matches_played': stats.matches_played,
                'win_rate': round(stats.win_rate * 100, 1),
                'goals_per_game': round(stats.goals_per_game, 2),
                'goals_against_per_game': round(stats.goals_against_per_game, 2),
                'form_points': stats.form_points,
                'recent_form': ''.join(stats.form[-5:]) if stats.form else 'N/A'
            }
        
        return analysis
    
    def get_training_report(self) -> Dict[str, Any]:
        """Eğitim raporunu döndürür."""
        report_path = self.models_dir / self.TRAINING_REPORT_FILENAME
        
        if report_path.exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return self._training_report
    
    def close(self) -> None:
        """Kaynakları temizler."""
        if self._owns_db and self.db_manager:
            self.db_manager.close()
            self.db_manager = None
    
    def __enter__(self) -> "ModelTrainer":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ============================================================================
# CLI ARAYÜZÜ
# ============================================================================

def main():
    """CLI giriş noktası"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="The Oracle - Model Eğitim Aracı"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Komutlar')
    
    # train komutu
    train_parser = subparsers.add_parser('train', help='Modelleri eğit')
    train_parser.add_argument(
        '--model', '-m',
        choices=['all', 'dixon-coles', 'xgboost'],
        default='all',
        help='Eğitilecek model'
    )
    
    # predict komutu
    predict_parser = subparsers.add_parser('predict', help='Tahmin yap')
    predict_parser.add_argument('home_team', type=str, help='Ev sahibi takım')
    predict_parser.add_argument('away_team', type=str, help='Deplasman takım')
    
    # rankings komutu
    subparsers.add_parser('rankings', help='Takım sıralamalarını göster')
    
    args = parser.parse_args()
    
    with ModelTrainer() as trainer:
        if args.command == 'train':
            if args.model == 'all':
                trainer.train_all()
            elif args.model == 'dixon-coles':
                trainer.train_dixon_coles()
            elif args.model == 'xgboost':
                trainer.train_xgboost()
        
        elif args.command == 'predict':
            trainer.load_models()
            result = trainer.predict_ensemble(args.home_team, args.away_team)
            
            print("\n" + "=" * 50)
            print(f"🏠 {args.home_team} vs {args.away_team} 🏃")
            print("=" * 50)
            print(f"\n📊 TAHMİN: {result['prediction_text']}")
            print(f"   Güven: {result['confidence']:.1%}")
            print(f"\n📈 Olasılıklar:")
            print(f"   Ev Kazanır:  {result['ensemble_probabilities']['home_win']:.1%}")
            print(f"   Beraberlik:  {result['ensemble_probabilities']['draw']:.1%}")
            print(f"   Dep Kazanır: {result['ensemble_probabilities']['away_win']:.1%}")
            print(f"\n⚽ Beklenen Goller:")
            print(f"   {args.home_team}: {result['expected_goals']['home']}")
            print(f"   {args.away_team}: {result['expected_goals']['away']}")
            print(f"   Toplam: {result['expected_goals']['total']}")
            print(f"\n🎯 En Olası Skorlar:")
            for score in result['most_likely_scores']:
                print(f"   {score['score']}: {score['probability']:.1%}")
            print("=" * 50 + "\n")
        
        elif args.command == 'rankings':
            trainer.load_models()
            
            if trainer.xgboost:
                print("\n📊 ELO SIRALAMASI")
                print("=" * 50)
                rankings = trainer.xgboost.get_team_elo_rankings()
                print(rankings.to_string())
                print("=" * 50 + "\n")
            
            if trainer.dixon_coles:
                print("\n⚡ TAKIM GÜÇLERİ (Dixon-Coles)")
                print("=" * 50)
                strengths = trainer.dixon_coles.get_team_strengths()
                print(strengths.to_string())
                print("=" * 50 + "\n")
        
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
