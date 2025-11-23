"""
Excel reporting utilities for greenhouse optimization project.

This module provides functions to export analysis results to Excel format
for easy viewing and sharing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import os
from datetime import datetime


def save_eda_to_excel(df: pd.DataFrame,
                      corr_matrix: pd.DataFrame,
                      filepath: str = 'rapor/01_eda_sonuclari.xlsx'):
    """
    Save exploratory data analysis results to Excel.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe
    corr_matrix : pd.DataFrame
        Correlation matrix
    filepath : str
        Path to save Excel file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: Summary Statistics
        summary_stats = df.describe()
        summary_stats.to_excel(writer, sheet_name='Özet İstatistikler', index=True)
        
        # Sheet 2: Correlation Matrix
        corr_matrix.to_excel(writer, sheet_name='Korelasyon Matrisi', index=True)
        
        # Sheet 3: Data Info
        info_data = {
            'Özellik': ['Satır Sayısı', 'Sütun Sayısı', 'Toplam Veri Noktası', 
                       'Eksik Değer Sayısı', 'Oluşturulma Tarihi'],
            'Değer': [
                len(df),
                len(df.columns),
                df.size,
                df.isna().sum().sum(),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name='Veri Bilgileri', index=False)
        
        # Sheet 4: First 1000 rows (sample)
        sample_df = df.head(1000)
        sample_df.to_excel(writer, sheet_name='Veri Örneği (İlk 1000)', index=True)
    
    print(f"EDA sonuçları Excel'e kaydedildi: {filepath}")


def save_sensitivity_to_excel(results: Dict,
                              filepath: str = 'rapor/02_co2_sensitivity_sonuclari.xlsx'):
    """
    Save CO₂ sensitivity analysis results to Excel.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_co2_sensitivity_analysis
    filepath : str
        Path to save Excel file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: Summary Results
        summary_data = {
            'Lambda (λ)': results['lambda_values'],
            'En İyi Enerji': results['energies'],
            'CO₂ Durumu': []
        }
        
        # Extract CO₂ states
        for config in results['configurations']:
            co2_state = 'AÇIK' if config.get('CO2', -1) == 1 else 'KAPALI'
            summary_data['CO₂ Durumu'].append(co2_state)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Özet Sonuçlar', index=False)
        
        # Sheet 2: Detailed Metadata
        metadata_list = []
        for meta in results['metadata']:
            metadata_list.append({
                'Lambda (λ)': meta['lambda'],
                'En İyi Enerji': meta['best_energy'],
                'Ortalama Enerji': meta['mean_energy'],
                'Standart Sapma': meta['std_energy'],
                'Süre (saniye)': meta['elapsed_time']
            })
        
        metadata_df = pd.DataFrame(metadata_list)
        metadata_df.to_excel(writer, sheet_name='Detaylı Metadatalar', index=False)
        
        # Sheet 3: Configurations
        config_data = []
        for i, (lam, config) in enumerate(zip(results['lambda_values'], results['configurations'])):
            row = {'Lambda (λ)': lam}
            for key, value in config.items():
                row[key] = 'AÇIK' if value == 1 else 'KAPALI'
            config_data.append(row)
        
        config_df = pd.DataFrame(config_data)
        config_df.to_excel(writer, sheet_name='Spin Konfigürasyonları', index=False)
        
        # Sheet 4: Analysis Info
        info_data = {
            'Bilgi': [
                'Analiz Tarihi',
                'Lambda Değerleri',
                'Toplam Deneme Sayısı',
                'En Düşük Enerji',
                'En Yüksek Enerji',
                'Enerji Aralığı'
            ],
            'Değer': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                ', '.join([str(lam) for lam in results['lambda_values']]),
                len(results['lambda_values']),
                min(results['energies']),
                max(results['energies']),
                max(results['energies']) - min(results['energies'])
            ]
        }
        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name='Analiz Bilgileri', index=False)
    
    print(f"CO₂ sensitivity sonuçları Excel'e kaydedildi: {filepath}")


def save_ablation_to_excel(results: Dict,
                           filepath: str = 'rapor/03_co2_ablation_sonuclari.xlsx'):
    """
    Save CO₂ ablation study results to Excel.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from run_co2_ablation_study
    filepath : str
        Path to save Excel file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: Comparison Summary
        comparison_data = {
            'Durum': ['Normal (CO₂ Aktif)', 'Ablation (CO₂ Zorla KAPALI)'],
            'Enerji': [
                results['normal']['energy'],
                results['ablation']['energy']
            ],
            'Ortalama Enerji': [
                results['normal']['metadata']['mean_energy'],
                results['ablation']['metadata']['mean_energy']
            ],
            'Standart Sapma': [
                results['normal']['metadata']['std_energy'],
                results['ablation']['metadata']['std_energy']
            ],
            'Süre (saniye)': [
                results['normal']['metadata']['elapsed_time'],
                results['ablation']['metadata']['elapsed_time']
            ],
            'CO₂ Durumu': [
                results['normal']['co2_state'],
                results['ablation']['co2_state']
            ]
        }
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_excel(writer, sheet_name='Karşılaştırma Özeti', index=False)
        
        # Sheet 2: Normal Configuration
        normal_config = results['normal']['configuration']
        normal_data = {
            'Kontrol Sistemi': list(normal_config.keys()),
            'Spin Değeri': list(normal_config.values()),
            'Durum': ['AÇIK' if v == 1 else 'KAPALI' for v in normal_config.values()]
        }
        normal_df = pd.DataFrame(normal_data)
        normal_df.to_excel(writer, sheet_name='Normal Konfigürasyon', index=False)
        
        # Sheet 3: Ablation Configuration
        ablation_config = results['ablation']['configuration']
        ablation_data = {
            'Kontrol Sistemi': list(ablation_config.keys()),
            'Spin Değeri': list(ablation_config.values()),
            'Durum': ['AÇIK' if v == 1 else 'KAPALI' for v in ablation_config.values()]
        }
        ablation_df = pd.DataFrame(ablation_data)
        ablation_df.to_excel(writer, sheet_name='Ablation Konfigürasyon', index=False)
        
        # Sheet 4: Difference Analysis
        diff_data = {
            'Metrik': [
                'Enerji Farkı',
                'Enerji Oranı',
                'CO₂ Penalty Ağırlığı',
                'Yüzde Değişim'
            ],
            'Değer': [
                results['comparison']['energy_difference'],
                results['comparison']['energy_ratio'],
                results['comparison']['co2_penalty'],
                ((results['ablation']['energy'] - results['normal']['energy']) / 
                 abs(results['normal']['energy']) * 100) if results['normal']['energy'] != 0 else 0
            ]
        }
        diff_df = pd.DataFrame(diff_data)
        diff_df.to_excel(writer, sheet_name='Fark Analizi', index=False)
        
        # Sheet 5: Analysis Info
        info_data = {
            'Bilgi': [
                'Analiz Tarihi',
                'Normal Durum Enerjisi',
                'Ablation Durum Enerjisi',
                'Enerji Farkı',
                'CO₂ Penalty'
            ],
            'Değer': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                results['normal']['energy'],
                results['ablation']['energy'],
                results['comparison']['energy_difference'],
                results['comparison']['co2_penalty']
            ]
        }
        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name='Analiz Bilgileri', index=False)
    
    print(f"CO₂ ablation sonuçları Excel'e kaydedildi: {filepath}")


def save_hamiltonian_to_excel(J: np.ndarray,
                             h: np.ndarray,
                             spin_names: List[str],
                             filepath: str = 'rapor/04_hamiltonian_model.xlsx'):
    """
    Save Hamiltonian model (J matrix and h vector) to Excel.
    
    Parameters:
    -----------
    J : np.ndarray
        Interaction matrix
    h : np.ndarray
        External field vector
    spin_names : list of str
        Names of spin variables
    filepath : str
        Path to save Excel file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Sheet 1: Interaction Matrix J
        J_df = pd.DataFrame(J, index=spin_names, columns=spin_names)
        J_df.to_excel(writer, sheet_name='Etkileşim Matrisi (J)', index=True)
        
        # Sheet 2: External Field h
        h_df = pd.DataFrame({
            'Kontrol Sistemi': spin_names,
            'Alan Değeri (h)': h
        })
        h_df.to_excel(writer, sheet_name='Dış Alan Vektörü (h)', index=False)
        
        # Sheet 3: Model Info
        info_data = {
            'Bilgi': [
                'Oluşturulma Tarihi',
                'Spin Sayısı',
                'Etkileşim Sayısı',
                'Maksimum J Değeri',
                'Minimum J Değeri',
                'Maksimum h Değeri',
                'Minimum h Değeri'
            ],
            'Değer': [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                len(spin_names),
                np.count_nonzero(J),
                np.max(J),
                np.min(J),
                np.max(h),
                np.min(h)
            ]
        }
        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name='Model Bilgileri', index=False)
    
    print(f"Hamiltonian model Excel'e kaydedildi: {filepath}")


def create_summary_report(eda_file: str = None,
                         sensitivity_file: str = None,
                         ablation_file: str = None,
                         hamiltonian_file: str = None,
                         filepath: str = 'rapor/00_ozet_rapor.xlsx'):
    """
    Create a summary report Excel file with links to other reports.
    
    Parameters:
    -----------
    eda_file : str, optional
        Path to EDA results file
    sensitivity_file : str, optional
        Path to sensitivity results file
    ablation_file : str, optional
        Path to ablation results file
    hamiltonian_file : str, optional
        Path to Hamiltonian model file
    filepath : str
        Path to save summary Excel file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    report_list = []
    if eda_file:
        report_list.append({
            'Rapor Adı': 'Keşifsel Veri Analizi',
            'Dosya Adı': os.path.basename(eda_file),
            'Açıklama': 'Korelasyon analizi ve istatistiksel özet'
        })
    if sensitivity_file:
        report_list.append({
            'Rapor Adı': 'CO₂ Duyarlılık Analizi',
            'Dosya Adı': os.path.basename(sensitivity_file),
            'Açıklama': 'Farklı lambda değerleri için optimizasyon sonuçları'
        })
    if ablation_file:
        report_list.append({
            'Rapor Adı': 'CO₂ Ablation Çalışması',
            'Dosya Adı': os.path.basename(ablation_file),
            'Açıklama': 'CO₂ zorla kapatıldığında enerji karşılaştırması'
        })
    if hamiltonian_file:
        report_list.append({
            'Rapor Adı': 'Hamiltonian Model',
            'Dosya Adı': os.path.basename(hamiltonian_file),
            'Açıklama': 'Ising model etkileşim matrisi ve dış alan vektörü'
        })
    
    summary_df = pd.DataFrame(report_list)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Rapor Listesi', index=False)
        
        # Add project info
        info_data = {
            'Bilgi': [
                'Proje Adı',
                'Oluşturulma Tarihi',
                'Rapor Sayısı'
            ],
            'Değer': [
                'Multi-Objective Climatic Optimisation of Agricultural Greenhouse Systems',
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                len(report_list)
            ]
        }
        info_df = pd.DataFrame(info_data)
        info_df.to_excel(writer, sheet_name='Proje Bilgileri', index=False)
    
    print(f"Özet rapor Excel'e kaydedildi: {filepath}")

