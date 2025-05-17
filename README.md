
# VehicleSegmentation_ML_Marketing
## 🚗 Canlı Demo

[🔗 Uygulamayı Görüntüle](vehiclesegmentation_ml_marketing ∙ main ∙ streamlit_app.py)


## 🔍 Project Overview / Proje Genel Bakış

**EN:**  
This project explores the use of unsupervised machine learning to segment vehicle customers in the automotive industry. By analyzing a variety of technical and financial car attributes, the goal is to discover distinct customer profiles. These insights can assist car manufacturers and dealers in making data-driven decisions regarding marketing campaigns, inventory planning, and product development.

**TR:**  
Bu proje, otomotiv sektöründe araç müşterilerinin segmentlere ayrılması için gözetimsiz makine öğrenmesi yöntemlerini kullanmaktadır. Araçların teknik ve finansal özelliklerini analiz ederek farklı müşteri profilleri ortaya çıkarılmak istenmiştir. Bu analizler, üreticilere ve bayilere pazarlama kampanyaları, stok planlaması ve ürün geliştirme konularında veri odaklı kararlar alma imkanı sağlar.

---

## 🎯 Objective / Proje Amacı

**EN:**  
To create meaningful clusters of vehicle models based on features such as price, engine size, fuel efficiency, weight, dimensions, horsepower, and more. The ultimate goal is to categorize customers or potential customer groups based on vehicle preferences, enabling targeted marketing strategies.

**TR:**  
Araçların fiyatı, motor hacmi, yakıt verimliliği, ağırlığı, boyutları, beygir gücü gibi özelliklerine göre anlamlı kümeler oluşturarak bu kümelerin hangi müşteri profillerini temsil ettiğini analiz etmek amaçlanmıştır. Böylece pazarlama stratejileri daha hedef odaklı hale getirilebilir.

---

## 📊 Dataset Summary / Veri Kümesi Özeti

- **Total Records / Toplam Kayıt**: 157 araç
- **Features / Özellik Sayısı**: 26 (20 sayısal, 6 kategorik)
- **Numerical Columns / Sayısal Sütunlar**: satış, fiyat, motor gücü, beygir gücü, uzunluk, genişlik, ağırlık, yakıt kapasitesi, MPG gibi teknik detaylar.
- **Categorical Columns / Kategorik Sütunlar**: marka, model, araç tipi gibi bilgiler.

### 🧼 Missing Data Handling / Eksik Veri İşleme

- **Resale columns**: %22,93 oranında eksik veri içeriyordu.
- Diğer eksik alanlar: fiyat, motor hacmi, beygir gücü, uzunluk vb.
- Sayısal sütunlarda ortanca (median), kategorik sütunlarda en sık değer (mode) ile doldurma işlemi uygulandı.

---

## ⚙️ Preprocessing Steps / Ön İşleme Aşamaları

- **Encoding**: Kategorik veriler etiketleme (label encoding) ile sayısallaştırıldı.
- **Feature Engineering**: Marka ve model birleşik bir değişken olarak yeniden oluşturuldu.
- **Outlier Detection / Aykırı Değer İşleme**: IQR yöntemi ile aykırı değerler sınırlandı.
- **Scaling / Ölçekleme**: MinMaxScaler kullanılarak tüm sayısal veriler [0,1] aralığına getirildi.

---

## 🔬 Dimensionality Reduction and Clustering / Boyut Azaltma ve Kümeleme

### PCA – Principal Component Analysis

**EN:** PCA was used to reduce the dataset to 2 principal components for better visualization and model performance. It also revealed the most influential features in the dataset.

**TR:** PCA ile veri 2 boyuta indirgenerek görselleştirme kolaylaştı ve algoritmaların daha etkili çalışması sağlandı. Aynı zamanda verideki en etkili değişkenler ortaya çıkarıldı.

---

## 🤖 Clustering Algorithms / Kümeleme Algoritmaları

### K-Means Clustering

- Amaç: Veriyi benzerliklerine göre k adet kümeye ayırmak
- Denenen parametreler: `k=2`'den `k=10`'a kadar farklı kümeler
- En iyi sonuç: `n_clusters=3`, `init='k-means++'`, `max_iter=10`

### Agglomerative Clustering

- Hiyerarşik kümeleme tekniği kullanıldı
- Mesafe ölçütleri: `euclidean`, `manhattan`, `cosine`
- Bağlantı yöntemleri: `ward`, `complete`, `average`, `single`
- En iyi sonuç: `n_clusters=3`, `metric='cosine'`, `linkage='complete'`

---

## 📈 Performance Metrics / Başarı Ölçütleri

| Metric / Metrik             | K-Means | Agglomerative |
|----------------------------|---------|---------------|
| Silhouette Score           | 0.5220  | 0.5203        |
| Calinski-Harabasz Index    | 217.47  | 213.17        |
| Davies-Bouldin Index       | 0.65    | 0.65          |
| Dunn Index                 | 0.92    | 0.90          |

**EN:** These results confirm that the clusters are well-separated and meaningful.

**TR:** Sonuçlar, kümelerin birbirinden belirgin şekilde ayrıldığını ve anlamlı olduğunu göstermektedir.

---

## 🔎 Interpretation of Segments / Kümelerin Yorumu

- **Segment 1 – Economy**: Düşük fiyatlı, yakıt verimliliği yüksek araçlar. Genellikle bütçe odaklı kullanıcılar için.
- **Segment 2 – Performance**: Yüksek beygir gücüne sahip, sportif modeller. Performans arayan kullanıcılar için.
- **Segment 3 – Luxury**: Büyük boyutlu, yüksek fiyatlı, konfor odaklı araçlar.

Bu segmentler üreticilere:
- Hedef kitle bazlı kampanyalar,
- Fiyatlandırma optimizasyonları,
- Ürün portföy planlaması
gibi stratejik avantajlar sunar.

---

## 📂 Project Structure / Proje Yapısı

```
Car-Segmentation-Project/
├── main.py
├── train_model.py
├── streamlit_app.py
├── requirements.txt
├── cleaned_clustered_data.xlsx
├── feature_names.pkl
├── scaler.pkl
├── kmeans_model.pkl
├── graphs/
├── sheets/
```

---

## ▶️ How to Run / Nasıl Çalıştırılır?

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Python Script
```bash
python main.py
```

### 3. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## ✅ Conclusion / Sonuç

**EN:**  
This project successfully integrates data preprocessing, clustering, and visualization to identify strategic vehicle customer segments. It can be adapted in real-world automotive marketing and business intelligence processes.

**TR:**  
Bu proje, veri ön işleme, kümeleme ve görselleştirme adımlarını başarılı şekilde birleştirerek otomotiv sektöründe stratejik müşteri segmentlerini ortaya koymuştur. Gerçek dünya uygulamalarına kolayca uyarlanabilir.
