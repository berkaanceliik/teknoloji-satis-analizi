# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)
pio.renderers.default = "notebook"

print("Kütüphaneler başarıyla yüklendi!")

# %% [markdown]
# # Teknoloji Mağazası Satış ve Müşteri Analiz Raporu
# 
# **Hazırlayan:** [Berkan Çelik]  
# **Fakülte:** İnsan ve Toplum Bilimleri Fakültesi  
# **Ders:** Veri Analizinde Bilgisayar Programlama 2 Final Ödevi  
# **Kaynak:** Sentetik Perakende Mağaza Verisi

# ## Özet ve Amaç
# Bu çalışma kapsamında, Türkiye'nin üç büyük şehrinde (İstanbul, Ankara, İzmir) faaliyet gösteren bir teknoloji mağaza zincirinin 300 satırlık müşteri harcama verisi incelenmiştir. Analizin amacı, müşteri demografisi (yaş, cinsiyet) ile alışveriş tercihleri (ürün kategorisi, harcama miktarı, ödeme yöntemi) arasındaki ilişkileri ortaya koyarak mağazanın satış stratejilerine katkı sağlamaktır.

# %%

df = pd.read_csv("data/veri_seti.csv")


print("--- Veri Setinin İlk 5 Satırı ---")
print(df.head())


print("\n--- Veri Boyutu ---")
print(f"Satır Sayısı: {df.shape[0]}, Sütun Sayısı: {df.shape[1]}")

# %% [markdown]
# ## 1. Tanımsal İstatistikler
# Bu bölümde veri setinde yer alan sayısal verilerin genel dağılımı ile kategorik değişkenlerin frekansları incelenmiştir.

# %%

print("--- Sayısal Değişkenlerin İstatistiki Özeti ---")
print(df.describe())


print("\n--- Şehirlere Göre Müşteri Dağılımı ---")
print(df["Sehir"].value_counts())


print("\n--- En Çok Tercih Edilen Ürün Kategorileri ---")
print(df["Urun_Kategorisi"].value_counts())


print("\n--- Ödeme Yöntemi Tercih Oranları (%) ---")
print(df["Odeme_Yontemi"].value_counts(normalize=True) * 100)

# %% [markdown]
# ## 2. Grafikler ile Görselleştirme ve Yorumlar
# Bu bölümde mağaza verileri,5 farklı grafik tipiyle görselleştirilmiş ve altlarına analiz yorumları eklenmiştir.

# %%
# GRAFİK 1: Yaş Dağılımı (Histogram)
plt.figure(figsize=(8, 4))
df["Yas"].hist(bins=15, color="skyblue", edgecolor="black")
plt.title("Müşterilerin Yaş Dağılımı (Histogram)")
plt.xlabel("Yaş")
plt.ylabel("Müşteri Sayısı")
plt.show()

# GRAFİK 2: Toplam Harcama Dağılımı (Boxplot)
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Toplam_Harcama"], color="lightgreen")
plt.title("Toplam Harcama Dağılımı (Boxplot)")
plt.xlabel("Harcama Tutarı (TL)")
plt.show()

# GRAFİK 3: Şehirlere Göre Ortalama Müşteri Memnuniyeti (Bar Plot)
plt.figure(figsize=(8, 4))
df.groupby("Sehir")["Musteri_Memnuniyeti"].mean().plot(kind="bar", color=["coral", "gold", "orchid"])
plt.title("Şehirlere Göre Ortalama Müşteri Memnuniyeti")
plt.ylabel("Ortalama Skor (1-5)")
plt.xticks(rotation=0)
plt.show()

# GRAFİK 4: Sayısal Değişkenlerin Korelasyon Matrisi (Heatmap)
plt.figure(figsize=(8, 5))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Değişkenler Arası Korelasyon Matrisi")
plt.show()

# GRAFİK 5: Yaş ve Toplam Harcama İlişkisi (Etkileşimli Scatter - Plotly)
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "vscode"

fig = px.scatter(df, 
                 x="Yas", 
                 y="Toplam_Harcama", 
                 color="Urun_Kategorisi",
                 title="Musteri Yasi ve Toplam Harcama Iliskisi (Etkilesimli Plotly)",
                 labels={"Yas": "Musteri Yasi", "Toplam_Harcama": "Harcama Tutari (TL)"})

fig.show()

# %% [markdown]
# ### Grafik Yorumları ve Bulgular
# 
# * **Grafik 1 (Histogram) Yorumu:** Mağazadan alışveriş yapan müşterilerin yaş dağılımı incelendiğinde, genç ve orta yaş grubunun yoğunlukta olduğu görülmektedir[cite: 28].
# * **Grafik 2 (Boxplot) Yorumu:** Harcama kutu grafiği, harcamaların genel olarak belirli bir bantta toplandığını ancak yüksek fiyatlı Bilgisayar ve Beyaz Eşya gibi ürünler sebebiyle sağa doğru uzayan bir dağılım olduğunu göstermektedir[cite: 29].
# * **Grafik 3 (Bar Plot) Yorumu:** Şehir bazlı memnuniyet oranlarında İstanbul, Ankara ve İzmir şubelerinin birbirine oldukça yakın ve yüksek (4.00 üzeri) skorlar aldığı gözlemlenmiştir.
# * **Grafik 4 (Heatmap) Yorumu:** Korelasyon matrisinde Yaş ile Memnuniyeti arasında güçlü bir doğrusal bağ olmasa da, Birim Fiyat ile Toplam Harcama arasında beklendiği gibi pozitif ve çok güçlü bir ilişki vardır.
# * **Grafik 5 (Scatter) Yorumu:** Etkileşimli grafikte görüldüğü üzere, yüksek bütçeli harcamalar (Bilgisayar ve Beyaz Eşya) her yaş grubundan müşteri tarafından yapılırken, Aksesuar ve Tablet gibi ürünler daha düşük bütçeli harcamaları oluşturmaktadır[cite: 23].
# 
# ### 📌 Önemli Sonuçlar (Yönetici Özeti)
# 1. Mağazada en yüksek ciro getiren ve birim fiyatı en yüksek olan ürün grubu **Bilgisayar** ve **Beyaz Eşya** kategorileridir.
# 2. Müşterilerin en çok kullandığı ödeme yöntemi **Kredi Kartı** ve **E-Cüzdan** uygulamalarıdır; nakit kullanımı daha arka plandadır.
# 3. Genel müşteri memnuniyeti şubeler genelinde oldukça yüksek olup, sadık bir müşteri kitlesine işaret etmektedir.


