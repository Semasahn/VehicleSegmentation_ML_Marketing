import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle

# 1. Veriyi oku
df = pd.read_excel(r"C:\Users\semas\OneDrive\Desktop\Car-Segmentation-Project-master\2.xlsx")

# 2. Sütun türlerini ayır
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# 3. Eksik değerleri doldur
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 4. Aykırı değerleri IQR ile kırp
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)
IQR = Q3 - Q1
for col in num_cols:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    df[col] = df[col].clip(lower, upper)

# 5. Ölçekleme (0-1 arası)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[num_cols])

# 6. KMeans eğit
model = KMeans(n_clusters=3, random_state=42)
model.fit(X_scaled)

# 7. Kaydet
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 8. Etiketli veriyi de kaydet
df["Cluster"] = model.predict(X_scaled)
df.to_excel("cleaned_clustered_data.xlsx", index=False)

print("✅ Tüm veriler hazırlandı ve model kaydedildi.")
