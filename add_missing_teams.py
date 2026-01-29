import os
import json
import pandas as pd

# Dosya yolları
RAW_DATA_PATH = "data/raw_csv"
MAPPING_FILE = "backend/core/team_mapping.json"

def update_team_mapping():
    # Mevcut mapping'i yükle
    try:
        with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    except FileNotFoundError:
        mapping = {}

    known_teams = set(mapping.keys())
    # Alias'ları da bilinenlere ekle (zaten varsa eklemeyelim diye)
    for aliases in mapping.values():
        for alias in aliases:
            known_teams.add(alias)

    new_teams_count = 0
    all_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')]

    print(f"📂 {len(all_files)} adet CSV dosyası taranıyor...")

    for filename in all_files:
        try:
            file_path = os.path.join(RAW_DATA_PATH, filename)
            # Encoding hatalarını önlemek için latin-1 veya utf-8 dene
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')

            # HomeTeam ve AwayTeam sütunlarını birleştir
            teams_in_file = pd.concat([df['HomeTeam'], df['AwayTeam']]).dropna().unique()

            for team in teams_in_file:
                team = team.strip()
                # Eğer takım mapping'de yoksa (ve aliaslarda da yoksa)
                if team not in known_teams:
                    # Yeni takım olarak ekle (Kendisini canonical isim yap)
                    mapping[team] = [team]
                    known_teams.add(team)
                    print(f"➕ Yeni Takım Eklendi: {team}")
                    new_teams_count += 1

        except Exception as e:
            print(f"⚠️ Hata ({filename}): {e}")

    # Güncellenmiş mapping'i kaydet
    if new_teams_count > 0:
        with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, indent=4, ensure_ascii=False)
        print(f"\n✅ Toplam {new_teams_count} yeni takım eklendi ve kaydedildi.")
    else:
        print("\n✨ Tüm takımlar zaten kayıtlı, eksik yok.")

if __name__ == "__main__":
    update_team_mapping()