"""
Master Script - Tüm Analizleri Adım Adım Çalıştır

Bu script tüm analizleri sırayla çalıştırır:
1. Keşifsel Veri Analizi (EDA)
2. Quantum Ising Model (QIM) Oluşturma
3. CO₂ Sensitivity ve Ablation Deneyleri
"""

import sys
import os
import time
from datetime import datetime

# Proje kök dizinine ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title: str):
    """Başlık yazdır."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_step(step_num: int, step_name: str):
    """Adım bilgisi yazdır."""
    print(f"\n{'='*70}")
    print(f"ADIM {step_num}: {step_name}")
    print(f"{'='*70}\n")


def run_script(script_name: str, description: str):
    """
    Bir Python script'ini çalıştır.
    
    Parameters:
    -----------
    script_name : str
        Çalıştırılacak script dosyası
    description : str
        Script açıklaması
    """
    print(f"Çalıştırılıyor: {script_name}")
    print(f"Açıklama: {description}")
    print("-" * 70)
    
    start_time = time.time()
    
    try:
        # Script'i modül olarak import et ve main fonksiyonunu çalıştır
        module_name = script_name.replace('.py', '').replace('/', '.').replace('\\', '.')
        
        if module_name.startswith('.'):
            module_name = module_name[1:]
        
        # Import ve çalıştır
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, script_name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'main'):
            module.main()
        else:
            print(f"UYARI: {script_name} içinde 'main' fonksiyonu bulunamadı.")
            print("Script doğrudan çalıştırılıyor...")
            exec(open(script_name).read())
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {script_name} başarıyla tamamlandı ({elapsed_time:.2f} saniye)")
        return True
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ HATA: {script_name} çalıştırılırken hata oluştu:")
        print(f"   {str(e)}")
        print(f"   Süre: {elapsed_time:.2f} saniye")
        return False


def main():
    """Ana fonksiyon - tüm adımları sırayla çalıştır."""
    
    print_header("GREENHOUSE QIM OPTIMIZATION - TÜM ANALİZLER")
    print(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Çalıştırılacak script'ler ve açıklamaları
    scripts = [
        {
            'file': '01_exploratory_analysis.py',
            'description': 'Keşifsel Veri Analizi - Veri yükleme, ön işleme, korelasyon analizi'
        },
        {
            'file': '02_build_qim_model.py',
            'description': 'Quantum Ising Model Oluşturma - Hamiltonian yapısı ve görselleştirme'
        },
        {
            'file': '03_experiments_co2_tests.py',
            'description': 'CO₂ Deneyleri - Sensitivity analizi ve ablation çalışması'
        }
    ]
    
    results = []
    total_start_time = time.time()
    
    # Her script'i sırayla çalıştır
    for i, script_info in enumerate(scripts, 1):
        print_step(i, script_info['description'])
        
        success = run_script(script_info['file'], script_info['description'])
        results.append({
            'step': i,
            'script': script_info['file'],
            'success': success
        })
        
        if not success:
            print(f"\n⚠ UYARI: Adım {i} başarısız oldu, ancak devam ediliyor...")
            response = input("\nDevam etmek istiyor musunuz? (e/h): ")
            if response.lower() != 'e':
                print("\nİşlem kullanıcı tarafından durduruldu.")
                break
    
    # Özet
    total_elapsed = time.time() - total_start_time
    print_header("ÇALIŞTIRMA ÖZETİ")
    
    print("Adım Sonuçları:")
    print("-" * 70)
    for result in results:
        status = "✓ BAŞARILI" if result['success'] else "✗ BAŞARISIZ"
        print(f"  Adım {result['step']}: {result['script']:35s} - {status}")
    
    print("\n" + "-" * 70)
    print(f"Toplam Süre: {total_elapsed:.2f} saniye ({total_elapsed/60:.2f} dakika)")
    print(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sonuç klasörleri kontrolü
    print("\n" + "-" * 70)
    print("Oluşturulan Dosyalar:")
    print("-" * 70)
    
    folders = {
        'figures/': 'Görseller',
        'rapor/': 'Excel Raporları',
        'results/': 'JSON Sonuçları'
    }
    
    for folder, description in folders.items():
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            print(f"  {description:20s} ({folder:15s}): {len(files)} dosya")
            if files:
                for file in files[:3]:  # İlk 3 dosyayı göster
                    print(f"    - {file}")
                if len(files) > 3:
                    print(f"    ... ve {len(files) - 3} dosya daha")
        else:
            print(f"  {description:20s} ({folder:15s}): Klasör bulunamadı")
    
    print("\n" + "=" * 70)
    print("TÜM ANALİZLER TAMAMLANDI!")
    print("=" * 70)
    print("\nSonuçları kontrol edin:")
    print("  - Excel raporları: rapor/ klasörü")
    print("  - Görseller: figures/ klasörü")
    print("  - JSON sonuçları: results/ klasörü")
    print()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nİşlem kullanıcı tarafından durduruldu (Ctrl+C).")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nKRİTİK HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

