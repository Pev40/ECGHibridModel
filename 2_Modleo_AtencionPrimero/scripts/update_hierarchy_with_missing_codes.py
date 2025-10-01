#!/usr/bin/env python3
"""
Script para actualizar la jerarquía con los códigos faltantes de ConditionNames_SNOMED-CT.csv
"""

import json
import pandas as pd
import os
from collections import defaultdict

def load_current_hierarchy(json_path):
    """Cargar jerarquía actual"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_condition_names(csv_path):
    """Cargar códigos SNOMED del archivo CSV"""
    df = pd.read_csv(csv_path)
    return df

def categorize_missing_codes(missing_df):
    """Categorizar códigos faltantes en grupos lógicos"""
    categories = {
        'rhythm': [],
        'conduction': [],
        'st_t_ischemia_mi': [],
        'hypertrophy': [],
        'ectopy': [],
        'axis_rotation_voltage': [],
        'other': []
    }
    
    # Mapeo de patrones de códigos a categorías
    for _, row in missing_df.iterrows():
        code = str(row['Snomed_CT'])
        acronym = row['Acronym Name']
        full_name = row['Full Name'].lower()
        
        # Categorización basada en patrones
        if any(keyword in full_name for keyword in ['rhythm', 'tachycardia', 'bradycardia', 'fibrillation', 'flutter', 'escape', 'arrest', 'block']):
            if any(keyword in full_name for keyword in ['ventricular', 'vt', 'vf', 'vfl', 'ver', 'aivr']):
                categories['rhythm'].append(code)
            elif any(keyword in full_name for keyword in ['atrial', 'afib', 'af', 'at', 'aec', 'eat']):
                categories['rhythm'].append(code)
            elif any(keyword in full_name for keyword in ['junctional', 'jt', 'ajr', 'jeb', 'jpt']):
                categories['rhythm'].append(code)
            elif any(keyword in full_name for keyword in ['sinus', 'sb', 'sr', 'st', 'sa', 'sar']):
                categories['rhythm'].append(code)
            else:
                categories['rhythm'].append(code)
        
        elif any(keyword in full_name for keyword in ['bundle', 'block', 'avb', 'lbbb', 'rbbb', 'fascicular', 'pfb', 'lafb', 'lpfb']):
            categories['conduction'].append(code)
        
        elif any(keyword in full_name for keyword in ['st', 't wave', 'q wave', 'mi', 'infarction', 'ischemia', 'ste', 'stdd', 'sttc', 'sttu', 'aqw', 'twe', 'two']):
            categories['st_t_ischemia_mi'].append(code)
        
        elif any(keyword in full_name for keyword in ['hypertrophy', 'enlargement', 'lvh', 'rvh', 'rah', 'lah', 'lae', 'rae']):
            categories['hypertrophy'].append(code)
        
        elif any(keyword in full_name for keyword in ['premature', 'ectopic', 'bigeminy', 'trigeminy', 'vpb', 'apb', 'svpb', 'vpac', 'bpac']):
            categories['ectopy'].append(code)
        
        elif any(keyword in full_name for keyword in ['axis', 'rotation', 'voltage', 'qrs', 'als', 'ars', 'ccr', 'cr', 'lvqrs', 'fqrs']):
            categories['axis_rotation_voltage'].append(code)
        
        else:
            categories['other'].append(code)
    
    return categories

def update_hierarchy(hierarchy, missing_categories):
    """Actualizar la jerarquía con los códigos faltantes"""
    # Agregar códigos faltantes a fine_codes
    new_fine_codes = []
    for category, codes in missing_categories.items():
        new_fine_codes.extend(codes)
    
    # Combinar con códigos existentes y eliminar duplicados
    all_fine_codes = list(set(hierarchy['fine_codes'] + new_fine_codes))
    hierarchy['fine_codes'] = all_fine_codes
    
    # Agregar códigos faltantes a coarse_groups
    for category, codes in missing_categories.items():
        if codes:  # Solo agregar si hay códigos
            if category in hierarchy['coarse_groups']:
                # Combinar con códigos existentes y eliminar duplicados
                existing_codes = set(hierarchy['coarse_groups'][category])
                new_codes = set(codes)
                hierarchy['coarse_groups'][category] = list(existing_codes | new_codes)
            else:
                hierarchy['coarse_groups'][category] = codes
    
    return hierarchy

def save_updated_hierarchy(hierarchy, output_path):
    """Guardar jerarquía actualizada"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, indent=2, ensure_ascii=False)

def main():
    # Rutas de archivos
    csv_path = 'datos/12Large/ConditionNames_SNOMED-CT.csv'
    json_path = 'datos/12Large/labels_hierarchy.json'
    backup_path = 'datos/12Large/labels_hierarchy_backup.json'
    
    print("=== ACTUALIZACIÓN DE JERARQUÍA CON CÓDIGOS FALTANTES ===\n")
    
    # 1. Crear backup
    print("1. Creando backup de jerarquía actual...")
    hierarchy = load_current_hierarchy(json_path)
    save_updated_hierarchy(hierarchy, backup_path)
    print(f"   Backup guardado en: {backup_path}")
    
    # 2. Cargar códigos faltantes
    print("\n2. Identificando códigos faltantes...")
    condition_df = load_condition_names(csv_path)
    current_fine_codes = set(hierarchy['fine_codes'])
    current_coarse_codes = set()
    for codes in hierarchy['coarse_groups'].values():
        current_coarse_codes.update(codes)
    all_current_codes = current_fine_codes | current_coarse_codes
    
    missing_df = condition_df[~condition_df['Snomed_CT'].astype(str).isin(all_current_codes)]
    print(f"   Códigos faltantes encontrados: {len(missing_df)}")
    
    # 3. Categorizar códigos faltantes
    print("\n3. Categorizando códigos faltantes...")
    missing_categories = categorize_missing_codes(missing_df)
    
    for category, codes in missing_categories.items():
        if codes:
            print(f"   - {category}: {len(codes)} códigos")
    
    # 4. Actualizar jerarquía
    print("\n4. Actualizando jerarquía...")
    updated_hierarchy = update_hierarchy(hierarchy, missing_categories)
    
    # 5. Guardar jerarquía actualizada
    print("\n5. Guardando jerarquía actualizada...")
    save_updated_hierarchy(updated_hierarchy, json_path)
    
    # 6. Resumen final
    print(f"\n6. RESUMEN DE ACTUALIZACIÓN:")
    print(f"   - Códigos fine originales: {len(hierarchy['fine_codes'])}")
    print(f"   - Códigos fine actualizados: {len(updated_hierarchy['fine_codes'])}")
    print(f"   - Nuevos códigos agregados: {len(updated_hierarchy['fine_codes']) - len(hierarchy['fine_codes'])}")
    
    print(f"\n   - Grupos coarse actualizados:")
    for group_name, codes in updated_hierarchy['coarse_groups'].items():
        print(f"     * {group_name}: {len(codes)} códigos")
    
    print(f"\n✅ Jerarquía actualizada exitosamente!")
    print(f"📁 Backup guardado en: {backup_path}")
    print(f"📁 Jerarquía actualizada en: {json_path}")

if __name__ == "__main__":
    main()
