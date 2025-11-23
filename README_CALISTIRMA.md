# Çalıştırma Rehberi

Bu rehber, projeyi adım adım nasıl çalıştıracağınızı açıklar.

## Hızlı Başlangıç (Tüm Analizleri Otomatik Çalıştır)

Tüm analizleri sırayla çalıştırmak için:

```bash
python 00_run_all.py
```

Bu script şu adımları sırayla çalıştırır:
1. Keşifsel Veri Analizi (EDA)
2. Quantum Ising Model Oluşturma
3. CO₂ Sensitivity ve Ablation Deneyleri

## Adım Adım Manuel Çalıştırma

Eğer analizleri tek tek çalıştırmak isterseniz:

### Adım 1: Keşifsel Veri Analizi

```bash
python 01_exploratory_analysis.py
```

**Ne yapar:**
- `data/greenhouse_data.xlsx` dosyasını yükler
- Veriyi temizler ve ön işler
- Korelasyon analizi yapar
- Scatter plot'lar oluşturur
- Excel raporu kaydeder: `rapor/01_eda_sonuclari.xlsx`

**Çıktılar:**
- `figures/` klasöründe görseller
- `rapor/01_eda_sonuclari.xlsx` Excel raporu

---

### Adım 2: Quantum Ising Model Oluşturma

```bash
python 02_build_qim_model.py
```

**Ne yapar:**
- Ising Hamiltonian'ı oluşturur
- Etkileşim matrisi (J) ve dış alan vektörü (h) hesaplar
- Modeli görselleştirir
- Excel raporu kaydeder: `rapor/04_hamiltonian_model.xlsx`

**Çıktılar:**
- `figures/interaction_matrix.png`
- `figures/external_field.png`
- `rapor/04_hamiltonian_model.xlsx` Excel raporu

---

### Adım 3: CO₂ Deneyleri

```bash
python 03_experiments_co2_tests.py
```

**Ne yapar:**
- CO₂ sensitivity analizi (lambda = [0.0, 0.25, 0.5, 1.0])
- CO₂ ablation çalışması (CO₂ zorla kapatıldığında)
- Sonuçları görselleştirir
- Excel raporları kaydeder
- Özet rapor oluşturur

**Çıktılar:**
- `figures/co2_sensitivity.png`
- `figures/co2_ablation.png`
- `results/co2_sensitivity.json`
- `results/co2_ablation.json`
- `rapor/02_co2_sensitivity_sonuclari.xlsx`
- `rapor/03_co2_ablation_sonuclari.xlsx`
- `rapor/00_ozet_rapor.xlsx` (özet rapor)

---

## Çalıştırma Sırası Önemli mi?

**Evet!** Script'ler sırayla çalıştırılmalı çünkü:

1. **01_exploratory_analysis.py** veriyi yükler ve temizler
2. **02_build_qim_model.py** optimizasyon modelini oluşturur
3. **03_experiments_co2_tests.py** modeli kullanarak deneyler yapar

Ancak, eğer önceki adımları zaten çalıştırdıysanız, sadece istediğiniz script'i çalıştırabilirsiniz.

## Sorun Giderme

### Hata: "Dataset not found"
- `data/greenhouse_data.xlsx` dosyasının `data/` klasöründe olduğundan emin olun

### Hata: "Module not found"
- Tüm bağımlılıkları yüklediğinizden emin olun:
  ```bash
  pip install -r requirements.txt
  ```

### Hata: "Permission denied" veya "File is locked"
- Excel dosyalarının açık olmadığından emin olun
- `rapor/` klasörüne yazma izniniz olduğundan emin olun

## Çıktı Dosyaları

Tüm çıktılar şu klasörlerde oluşturulur:

- **`rapor/`** - Excel raporları (Türkçe başlıklar)
- **`figures/`** - Görseller (PNG formatında)
- **`results/`** - JSON formatında ham sonuçlar

## Örnek Çalıştırma

```bash
# Tüm analizleri çalıştır
python 00_run_all.py

# Veya adım adım:
python 01_exploratory_analysis.py
python 02_build_qim_model.py
python 03_experiments_co2_tests.py
```

Her script çalıştıktan sonra ilgili klasörlerde çıktıları kontrol edebilirsiniz.

