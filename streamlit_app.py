import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Araç Segmentasyonu", layout="wide")
st.title("🚗 Araç Segmentasyonu Dashboard")

# Sekmeli üst menü
tab1, tab2, tab3, tab4 = st.tabs(["Veriyi İncele", "Görseller", "Segment Analizi", "Segment Tahmini"])

# 1️⃣ VERİ SETİ GÖSTER
with tab1:
    st.subheader("📄 Veri Seti")
    try:
        df = pd.read_excel("2.xlsx")
        st.dataframe(df, use_container_width=True)

        st.subheader("✅ Ön İşlenmiş Veri")
        df_clean = pd.read_excel("cleaned_clustered_data.xlsx")
        st.dataframe(df_clean, use_container_width=True)

    except FileNotFoundError:
        st.error("Veri dosyaları bulunamadı. Lütfen 2.xlsx ve cleaned_clustered_data.xlsx dosyalarını kontrol edin.")

# 2️⃣ GÖRSELLER
with tab2:
    st.subheader("📈 Görseller")
    try:
        st.image("graphs/step2_initial_data_distribution.png", caption="Başlangıç Veri Dağılımı")
        st.image("graphs/step3_before_cleaning_data_distribution.png", caption="Temizlenmeden Önce Veri Dağılımı")
        st.image("graphs/step3_cleaned_data_distribution.png", caption="Temizlenmiş Veri Dağılımı")
        st.image("graphs/step5_standardized_data_distribution.png", caption="Standardize Edilmiş Veri Dağılımı")
        st.image("graphs/step5_pca_components.png", caption="PCA Bileşenleri")
        st.image("graphs/step5_kmeans_clustering.png", caption="K-Means Kümeleme")
        st.image("graphs/step5_agglomerative_clustering.png", caption="Agglomerative Kümeleme")
        st.image("graphs/step6_clustering_visualization.png", caption="Kümeleme Sonuçları Karşılaştırması")

    except:
        st.warning("Görsel dosyalar eksik olabilir veya yol hatalı.")

# 3️⃣ SEGMENT ANALİZİ
with tab3:
    st.subheader("🔍 Segment Analizi")
    try:
        df_results = pd.read_excel("sheets/step6_final_results.xlsx")
        st.write("Her segmentin ortalama değerleri:")
        st.dataframe(df_results.groupby("Cluster_KMeans").mean(numeric_only=True))

        st.markdown("#### 🎯 Segment Açıklamaları")
        st.markdown("""
- **Segment 0**: Küçük motor hacmi, düşük beygir gücü, düşük fiyat
- **Segment 1**: Orta seviye aile araçları, dengeli fiyat ve performans
- **Segment 2**: Lüks segment, yüksek motor gücü ve yüksek fiyat
""")

        st.markdown("#### 🚗 Örnek Araçlar (Her Segmentten)")
        df_cleaned = pd.read_excel("cleaned_clustered_data.xlsx")
        for i in sorted(df_cleaned["Cluster"].unique()):
            st.markdown(f"**Segment {i}**")
            sample_df = df_cleaned[df_cleaned["Cluster"] == i][["manufact", "model", "price", "engine_s", "horsepow"]].sample(n=3, random_state=42)
            sample_df.columns = ["Marka", "Model", "Fiyat", "Motor Hacmi", "Beygir Gücü"]
            st.dataframe(sample_df)

    except Exception as e:
        st.error(f"Hata oluştu: {e}")

# 4️⃣ SEGMENT TAHMİNİ
with tab4:
    st.subheader("🔮 Yeni Araç Segmenti Tahmini")

    try:
        with open("feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)

        # Girdi formu dinamik olarak oluştur
        input_values = []
        for feature in feature_names:
            label_tr = feature.replace("_", " ").capitalize() + " (" + feature + ")"
            value = st.number_input(label_tr, min_value=0.0)
            input_values.append(value)

        if st.button("Tahmin Et"):
            # Model ve scaler'ı yükle
            scaler = pickle.load(open("scaler.pkl", "rb"))
            model = pickle.load(open("kmeans_model.pkl", "rb"))

            # DataFrame oluştur
            user_input = pd.DataFrame([input_values], columns=feature_names)
            user_scaled = scaler.transform(user_input)
            segment = model.predict(user_scaled)[0]

            # Açıklama
            if segment == 0:
                desc = """
### 🟢 Segment 0 — Ekonomik Araçlar

**Genel Özellikler:**
- Motor hacmi: 1.0 – 1.6 L
- Fiyat seviyesi: Düşük (300.000₺ altı)
- Yakıt tasarruflu, kompakt

**Hedef Kullanıcılar:**
- Öğrenciler, şehir içi kullanıcılar

**Örnek Modeller:**
- Fiat Egea, Hyundai i10, Renault Clio
"""
            elif segment == 1:
                desc = """
### ⚖️ Segment 1 — Orta Seviye Aile Araçları

**Genel Özellikler:**
- Motor hacmi: 1.6 - 2.0 L
- Fiyat: 300.000₺ - 800.000₺
- Dengeli yakıt ekonomisi ve performans

**Hedef Kullanıcılar:**
- Aileler, şirket personelleri

**Örnek Modeller:**
- Toyota Corolla, VW Golf, Renault Megane
"""
            else:
                desc = """
### 🔺 Segment 2 — Lüks & Performans Araçları

**Genel Özellikler:**
- Motor hacmi: Yüksek (2.0 L ve üzeri)
- Fiyat seviyesi: Yüksek (800.000₺ ve üstü)
- Yüksek beygir gücü, gelişmiş güvenlik ve konfor

**Hedef Kullanıcılar:**
- Yöneticiler, performans ve prestij arayanlar

**Örnek Modeller:**
- BMW 5 Serisi, Audi A6, Mercedes E-Class
"""

            st.success(f"✅ Bu araç muhtemelen Segment {segment}'e girer.")
            st.markdown(desc)

    except Exception as e:
        st.error(f"Hata oluştu: {e}")
