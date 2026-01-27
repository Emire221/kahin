# 📖 THE ORACLE - KULLANIM KILAVUZU

> Premier Lig Maç Tahmin ve Simülasyon Sistemi

---

## 📊 PROJE DURUM ÖZETİ

| Faz | Açıklama | Durum | Tamamlanma |
|-----|----------|-------|------------|
| FAZ 1 | Temel ve Veri (Data & Database) | ✅ Tamamlandı | %100 |
| FAZ 2 | Beyin ve Eğitim (Math & AI Models) | ✅ Tamamlandı | %100 |
| FAZ 3 | Denetim ve Simülasyon (Backtest) | ✅ Tamamlandı | %100 |
| FAZ 4 | Canlı Boru Hattı (API & Gemini) | ⏳ Bekliyor | %0 |
| FAZ 5 | Vitrin (Flutter Frontend) | ⏳ Bekliyor | %0 |

---

## 📁 GÜNCEL DOSYA YAPISI

```
c:\Users\ahmet\Desktop\Oracle\
│
├── 📄 requirements.txt              # Python bağımlılıkları
├── 📄 📖 KULLANIM KILAVUZU.md       # Bu dosya
├── 📄 📅 THE ORACLE... TAKVİMİ.txt  # 5 aşamalı plan
│
├── 📂 data/
│   ├── 📂 raw_csv/                  # API'den çekilen CSV'ler
│   │   ├── PL_2023_2024.csv
│   │   ├── PL_2024_2025.csv
│   │   └── PL_2025_2026.csv
│   ├── 📂 logs/                     # Backtest raporları
│   │   └── backtest_report_*.json
│   └── 🗄️ oracle.db                 # SQLite veritabanı (990 maç)
│
├── 📂 backend/
│   ├── 📄 __init__.py
│   ├── 📄 main.py                   # CLI giriş noktası
│   │
│   ├── 📂 core/                     # ✅ FAZ 1
│   │   ├── 📄 config.py             # Merkezi konfigürasyon
│   │   └── 📄 team_mapping.json     # Takım isim eşleştirmesi (35 takım)
│   │
│   ├── 📂 database/                 # ✅ FAZ 1
│   │   └── 📄 db_manager.py         # SQLite CRUD işlemleri
│   │
│   ├── 📂 data/                     # ✅ FAZ 1
│   │   ├── 📄 data_loader.py        # CSV yükleyici
│   │   └── 📄 api_fetcher.py        # Football-Data.org API
│   │
│   ├── 📂 models/                   # ✅ FAZ 2
│   │   ├── 📄 dixon_coles.py        # Poisson skor tahmini (~550 satır)
│   │   ├── 📄 xgboost_model.py      # ML sonuç tahmini (~580 satır)
│   │   ├── 🔮 dixon_coles.pkl       # Eğitilmiş model
│   │   ├── 🔮 xgboost.pkl           # Eğitilmiş model (4.8 MB)
│   │   └── 📄 training_report.json
│   │
│   ├── 📂 training/                 # ✅ FAZ 2
│   │   └── 📄 trainer.py            # Model eğitim yöneticisi
│   │
│   ├── 📂 simulation/               # ✅ FAZ 3
│   │   ├── 📄 wallet.py             # Sanal kasa (~350 satır)
│   │   └── 📄 backtest_engine.py    # Walk-Forward test (~650 satır)
│   │
│   ├── 📂 services/                 # ✅ FAZ 3
│   │   └── 📄 value_calculator.py   # Value bet hesaplayıcı (~400 satır)
│   │
│   ├── 📂 scrapers/                 # 📅 FAZ 4
│   └── 📂 api/                      # 📅 FAZ 4
│
└── 📂 frontend/                     # 📅 FAZ 5
```

---

## 🚀 KURULUM VE ÇALIŞTIRMA

### Ön Gereksinimler

- Python 3.10 veya üzeri
- pip (Python paket yöneticisi)

### Adım 1: Bağımlılıkları Yükle

```powershell
cd c:\Users\ahmet\Desktop\Oracle
pip install -r requirements.txt
```

### Adım 2: Veritabanını Oluştur

```powershell
& "C:\Users\ahmet\AppData\Local\Programs\Python\Python312\python.exe" -m backend.main init-db
```

### Adım 3: Veri Çek (API ile)

```powershell
# Football-data.org API anahtarınızla
& "C:\Users\ahmet\AppData\Local\Programs\Python\Python312\python.exe" -c "
from backend.data.api_fetcher import fetch_premier_league_data
fetch_premier_league_data('YOUR_API_KEY', years_back=5)
"
```

> **Not:** API anahtarı almak için: <https://www.football-data.org/client/register>

### Adım 4: Veri Durumunu Kontrol Et

```powershell
& "C:\Users\ahmet\AppData\Local\Programs\Python\Python312\python.exe" -m backend.main summary
```

Beklenen çıktı:

```
==================================================
THE ORACLE - VERİ ÖZETİ
==================================================
Toplam Maç: 990
Takım Sayısı: 25
Tarih Aralığı: 2023-08-11 - 2026-01-26

Sezon Dağılımı:
  2023-2024: 380 maç
  2024-2025: 380 maç
  2025-2026: 230 maç

✓ Veri tutarlılığı OK
==================================================
```

### Adım 5: Modelleri Eğit

```powershell
& "C:\Users\ahmet\AppData\Local\Programs\Python\Python312\python.exe" -m backend.training.trainer train
```

Bu komut:

1. Dixon-Coles modelini eğitir (Poisson tabanlı)
2. XGBoost modelini eğitir (19 öznitelik)
3. Modelleri `.pkl` dosyası olarak kaydeder

### Adım 6: Tahmin Yap

```powershell
& "C:\Users\ahmet\AppData\Local\Programs\Python\Python312\python.exe" -m backend.training.trainer predict "Arsenal" "Chelsea"
```

Örnek çıktı:

```
==================================================
🏠 Arsenal vs Chelsea 🏃
==================================================

📊 TAHMİN: Arsenal Kazanır
   Güven: 57.8%

📈 Olasılıklar:
   Ev Kazanır:  57.8%
   Beraberlik:  23.9%
   Dep Kazanır: 18.3%

⚽ Beklenen Goller:
   Arsenal: 1.76
   Chelsea: 0.72
   Toplam: 2.49

🎯 En Olası Skorlar:
   1-0: 13.5%
   2-0: 12.9%
   1-1: 11.8%
==================================================
```

### Adım 7: Backtest Çalıştır

```powershell
& "C:\Users\ahmet\AppData\Local\Programs\Python\Python312\python.exe" -m backend.simulation.backtest_engine --export
```

Bu komut:

1. Walk-Forward Validation uygular
2. Her sezon için eğit-test döngüsü çalıştırır
3. Value bet stratejisini simüle eder
4. Sonuçları JSON olarak kaydeder

---

## 🧠 MODEL AÇIKLAMALARI

### 1. Dixon-Coles Modeli

**Dosya:** `backend/models/dixon_coles.py`

Poisson dağılımı tabanlı skor tahmin modeli.

**Özellikler:**

- Takım hücum/savunma güç parametreleri
- İç saha avantajı faktörü
- Düşük skorlu maç düzeltmesi (rho)
- Zaman bazlı ağırlıklama (yakın maçlar daha önemli)

**Çıktılar:**

- Maç sonucu olasılıkları (1-X-2)
- Beklenen gol sayıları
- Alt/Üst 2.5 olasılıkları
- KG Var/Yok olasılıkları
- En olası skorlar

### 2. XGBoost Modeli

**Dosya:** `backend/models/xgboost_model.py`

Gradient Boosting tabanlı maç sonucu tahmincisi.

**19 Öznitelik:**

| # | Öznitelik | Açıklama |
|---|-----------|----------|
| 1 | home_elo | Ev sahibi Elo rating |
| 2 | away_elo | Deplasman Elo rating |
| 3 | elo_diff | Elo farkı |
| 4 | home_form | Ev sahibi son 5 maç puanı |
| 5 | away_form | Deplasman son 5 maç puanı |
| 6 | form_diff | Form farkı |
| 7 | home_goals_scored_avg | Ev sahibi gol ortalaması |
| 8 | away_goals_scored_avg | Deplasman gol ortalaması |
| 9 | home_goals_conceded_avg | Ev sahibi yenilen gol ort. |
| 10 | away_goals_conceded_avg | Deplasman yenilen gol ort. |
| 11 | home_win_rate | Ev sahibi kazanma oranı |
| 12 | away_win_rate | Deplasman kazanma oranı |
| 13 | h2h_home_wins | H2H ev sahibi galibiyetleri |
| 14 | h2h_away_wins | H2H deplasman galibiyetleri |
| 15 | h2h_draws | H2H beraberlikler |
| 16 | home_home_form | Ev sahibinin evdeki formu |
| 17 | away_away_form | Deplasmanın dışarıdaki formu |
| 18 | home_days_rest | Ev sahibi dinlenme günü |
| 19 | away_days_rest | Deplasman dinlenme günü |

### 3. Ensemble Sistem

İki modelin ağırlıklı ortalaması:

- **Dixon-Coles:** %40
- **XGBoost:** %60

---

## 📊 BACKTEST SONUÇLARI

### Walk-Forward Validation (min_train_seasons=1)

| Test Sezonu | Eğitim Verisi | Bahis | Kazanç | Kayıp | ROI |
|-------------|---------------|-------|--------|-------|-----|
| 2024-2025 | 2023-24 | 63 | 41 | 22 | -1.77% |
| 2025-2026 | 2023-24 + 2024-25 | 29 | 24 | 5 | +26.91% |
| **TOPLAM** | | **92** | **65** | **27** | **+7.27%** |

### Performans Metrikleri

| Metrik | Sonuç | Hedef | Durum |
|--------|-------|-------|-------|
| ROI | +7.27% | > %5 | ✅ |
| Hit Rate | 70.65% | > %55 | ✅ |
| Max Drawdown | 12.06% | < %20 | ✅ |

### Kasa Simülasyonu

- Başlangıç: 1000 birim
- Final: 1066.84 birim
- Net Kar: +66.84 birim

---

## 💡 VALUE BET STRATEJİSİ

### Formül

```
Expected Value (EV) = Model Olasılığı × Bahis Oranı

Value Bet = EV > 1.05 VE Edge > %3
```

### Örnek

```
Model Arsenal kazanır diyor: %65 olasılık
Bahisçi oranı: 1.75

EV = 0.65 × 1.75 = 1.1375
Edge = 0.65 - (1/1.75) = 0.65 - 0.57 = 0.08 (%8)

Sonuç: VALUE BET ✅ (EV > 1.05 ve Edge > %3)
```

---

## 📂 RAPOR DOSYALARI

Backtest raporları `data/logs/` klasörüne kaydedilir:

```json
{
  "config": {
    "initial_balance": 1000,
    "stake": 10,
    "value_threshold": 1.05,
    "min_edge": 0.03,
    "dixon_weight": 0.4,
    "xgboost_weight": 0.6
  },
  "summary": {
    "total_bets": 92,
    "total_wins": 65,
    "roi": 7.27,
    "hit_rate": 70.65,
    "max_drawdown": 12.06
  },
  "season_results": [...],
  "transactions": [...],
  "balance_history": [...]
}
```

---

## 🔧 GELİŞTİRİLECEK MODÜLLER (FAZ 4 & 5)

### FAZ 4: Canlı Boru Hattı

- [ ] `scrapers/fixture_bot.py` - Gelecek maçları çekme
- [ ] `scrapers/news_bot.py` - Spor haberlerini çekme
- [ ] `services/gemini_service.py` - Google AI ile haber yorumlama
- [ ] `api/main.py` - FastAPI REST endpointleri

### FAZ 5: Flutter Frontend

- [ ] Web arayüzü
- [ ] Canlı tahmin ekranı
- [ ] Backtest görselleştirme
- [ ] Kasa takip paneli

---

## ❓ SIK SORULAN SORULAR

### Python komutu çalışmıyor?

Windows'ta Python PATH'te olmayabilir. Tam yol kullanın:

```powershell
& "C:\Users\ahmet\AppData\Local\Programs\Python\Python312\python.exe" -m ...
```

### "Bilinmeyen takım" uyarısı alıyorum?

Yeni sezona yükselen takımlar eğitim verisinde olmayabilir. `team_mapping.json` dosyasına ekleyebilirsiniz.

### Daha eski sezonları nasıl eklerim?

Football-data.org Free API'de sadece son 3-4 sezon var. Daha eski veriler için:

1. <https://www.football-data.co.uk/englandm.php> adresinden CSV indirin
2. `data/raw_csv/` klasörüne koyun
3. `python -m backend.main load-data` çalıştırın

### Model performansı kötü?

- Daha fazla eğitim verisi ekleyin (en az 3-5 sezon önerilir)
- `value_threshold` değerini artırın (1.10 gibi)
- `min_edge` değerini artırın (%5 gibi)

---

## 📞 KOMUT REFERANSİ

| Komut | Açıklama |
|-------|----------|
| `python -m backend.main init-db` | Veritabanı oluştur |
| `python -m backend.main load-data` | CSV'leri yükle |
| `python -m backend.main summary` | Veri özeti |
| `python -m backend.training.trainer train` | Modelleri eğit |
| `python -m backend.training.trainer predict "Takım1" "Takım2"` | Tahmin yap |
| `python -m backend.training.trainer rankings` | Sıralamalar |
| `python -m backend.simulation.backtest_engine --export` | Backtest çalıştır |

---

*Son Güncelleme: 2026-01-28 01:46*
*Versiyon: 0.3.0 (FAZ 1-2-3 Tamamlandı)*
