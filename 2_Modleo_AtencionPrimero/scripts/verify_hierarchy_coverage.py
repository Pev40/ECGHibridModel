#!/usr/bin/env python3
"""
Script para verificar que todos los códigos de ConditionNames_SNOMED-CT.csv
estén incluidos en la jerarquía labels_hierarchy.json
"""

import json
import pandas as pd
import os

def load_condition_names(csv_path):
    """Cargar códigos SNOMED del archivo CSV"""
    df = pd.read_csv(csv_path)
    snomed_codes = df['Snomed_CT'].astype(str).tolist()
    return snomed_codes, df

def load_hierarchy_codes(json_path):
    """Cargar códigos de la jerarquía"""
    with open(json_path, 'r', encoding='utf-8') as f:
        hierarchy = json.load(f)
    
    fine_codes = hierarchy.get('fine_codes', [])
    coarse_groups = hierarchy.get('coarse_groups', {})
    
    # Obtener todos los códigos únicos de coarse_groups
    all_hierarchy_codes = set(fine_codes)
    for group_codes in coarse_groups.values():
        all_hierarchy_codes.update(group_codes)
    
    return list(all_hierarchy_codes), fine_codes, coarse_groups

def compare_codes(condition_codes, hierarchy_codes, condition_df):
    """Comparar códigos y encontrar diferencias"""
    condition_set = set(condition_codes)
    hierarchy_set = set(hierarchy_codes)
    
    # Códigos en ConditionNames pero no en jerarquía
    missing_in_hierarchy = condition_set - hierarchy_set
    
    # Códigos en jerarquía pero no en ConditionNames
    missing_in_conditions = hierarchy_set - condition_set
    
    # Códigos comunes
    common_codes = condition_set & hierarchy_set
    
    return {
        'missing_in_hierarchy': missing_in_hierarchy,
        'missing_in_conditions': missing_in_conditions,
        'common_codes': common_codes,
        'total_condition_codes': len(condition_codes),
        'total_hierarchy_codes': len(hierarchy_codes),
        'coverage_percentage': len(common_codes) / len(condition_codes) * 100
    }

def get_condition_details(missing_codes, condition_df):
    """Obtener detalles de las condiciones faltantes"""
    details = []
    for code in missing_codes:
        row = condition_df[condition_df['Snomed_CT'].astype(str) == code]
        if not row.empty:
            details.append({
                'code': code,
                'acronym': row.iloc[0]['Acronym Name'],
                'full_name': row.iloc[0]['Full Name']
            })
    return details

def main():
    # Rutas de archivos
    csv_path = 'datos/12Large/ConditionNames_SNOMED-CT.csv'
    json_path = 'datos/12Large/labels_hierarchy.json'
    
    print("=== VERIFICACIÓN DE COBERTURA DE JERARQUÍA ===\n")
    
    # Cargar datos
    print("1. Cargando códigos de ConditionNames...")
    condition_codes, condition_df = load_condition_names(csv_path)
    print(f"   Total códigos en ConditionNames: {len(condition_codes)}")
    
    print("\n2. Cargando códigos de jerarquía...")
    hierarchy_codes, fine_codes, coarse_groups = load_hierarchy_codes(json_path)
    print(f"   Total códigos únicos en jerarquía: {len(hierarchy_codes)}")
    print(f"   Códigos fine: {len(fine_codes)}")
    print(f"   Grupos coarse: {len(coarse_groups)}")
    
    # Comparar
    print("\n3. Comparando códigos...")
    comparison = compare_codes(condition_codes, hierarchy_codes, condition_df)
    
    print(f"   Códigos comunes: {len(comparison['common_codes'])}")
    print(f"   Cobertura: {comparison['coverage_percentage']:.1f}%")
    
    # Mostrar códigos faltantes en jerarquía
    if comparison['missing_in_hierarchy']:
        print(f"\n4. CÓDIGOS FALTANTES EN JERARQUÍA ({len(comparison['missing_in_hierarchy'])}):")
        missing_details = get_condition_details(comparison['missing_in_hierarchy'], condition_df)
        for detail in missing_details:
            print(f"   - {detail['code']}: {detail['acronym']} - {detail['full_name']}")
    else:
        print("\n4. ✅ Todos los códigos de ConditionNames están en la jerarquía")
    
    # Mostrar códigos en jerarquía pero no en ConditionNames
    if comparison['missing_in_conditions']:
        print(f"\n5. CÓDIGOS EN JERARQUÍA PERO NO EN CONDITIONNAMES ({len(comparison['missing_in_conditions'])}):")
        for code in sorted(comparison['missing_in_conditions']):
            print(f"   - {code}")
    
    # Resumen de grupos coarse
    print(f"\n6. DISTRIBUCIÓN POR GRUPOS COARSE:")
    for group_name, codes in coarse_groups.items():
        print(f"   - {group_name}: {len(codes)} códigos")
    
    # Recomendaciones
    print(f"\n7. RECOMENDACIONES:")
    if comparison['coverage_percentage'] < 100:
        print(f"   ⚠️  Cobertura incompleta ({comparison['coverage_percentage']:.1f}%)")
        print(f"   📝 Se recomienda actualizar la jerarquía para incluir los {len(comparison['missing_in_hierarchy'])} códigos faltantes")
    else:
        print(f"   ✅ Cobertura completa (100%)")
        print(f"   🎉 Todos los códigos están correctamente incluidos en la jerarquía")

if __name__ == "__main__":
    main()
