import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Veri setini okuma
# Dosya data klasöründeyse oradan, değilse ana dizinden oku
if os.path.exists("data/spg.csv"):
    df = pd.read_csv("data/spg.csv")
else:
    df = pd.read_csv("spg.csv")

# Tarih sütunu işimize yaramayacak, siliyoruz ama index olarak tutabiliriz
# df.set_index('time', inplace=True) # Gerekirse açılabilir

print("Veri boyutu:", df.shape)

# Feature Engineering Kısmı
# Geçmiş saatlerin verilerini ekliyoruz (Lag Features)
target = "generated_power_kw"

# Son 3 saatin üretim verisini ekle
df['power_lag_1'] = df[target].shift(1)
df['power_lag_2'] = df[target].shift(2)
df['power_lag_3'] = df[target].shift(3)

# Radyasyon verisinin bir önceki saatini de ekleyelim
if 'shortwave_radiation_backwards_sfc' in df.columns:
    df['radiation_lag_1'] = df['shortwave_radiation_backwards_sfc'].shift(1)

# Shift işlemi yaptığımız için ilk satırlar boş (NaN) kalır, onları siliyoruz
df.dropna(inplace=True)

# Girdi (X) ve Çıktı (y) değişkenlerini ayarla
X = df.drop(columns=[target, "time"], errors='ignore')
y = df[target]

# Eğitim ve Test olarak ayır (%20 test)
# Shuffle=False yapıyoruz çünkü bu bir zaman serisi, sırayı bozmamalıyız
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Eğitim seti sayısı:", len(X_train))
print("Test seti sayısı:", len(X_test))

# Modeli Kurma
# Pipeline kullanıyoruz ki scaler ve model bir arada olsun
# ExtraTreesRegressor kullandık çünkü denemelerde en iyi sonucu bu verdi
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

print("Model eğitiliyor, lütfen bekleyin...")
pipeline.fit(X_train, y_train)

# Test seti üzerinde tahmin yap
y_pred = pipeline.predict(X_test)

# Sonuçları yazdır
score = r2_score(y_test, y_pred)
print("Model R2 Skoru:", score)

# Eğer skor iyiyse kaydet
if score > 0.80:
    # Klasör yoksa oluştur
    if not os.path.exists("models"):
        os.makedirs("models")

    # Modeli kaydet
    joblib.dump(pipeline, "models/solar_model.pkl")
    print("Model başarıyla models/solar_model.pkl dosyasına kaydedildi.")
else:
    print("Skor düşük olduğu için model kaydedilmedi.")