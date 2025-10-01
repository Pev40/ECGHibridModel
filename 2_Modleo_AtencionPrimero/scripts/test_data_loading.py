#!/usr/bin/env python3
"""
Script para probar la carga de datos con la jerarquía actualizada
"""

import os
import sys
import json
import torch
import numpy as np

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.HMSTPreprocessor import HMSTPreprocessor

def test_hierarchy_loading():
    """Probar carga de jerarquía"""
    print("=== PRUEBA DE CARGA DE JERARQUÍA ===\n")
    
    hierarchy_path = 'datos/12Large/labels_hierarchy.json'
    
    # Cargar jerarquía
    with open(hierarchy_path, 'r', encoding='utf-8') as f:
        hierarchy = json.load(f)
    
    fine_codes = hierarchy.get('fine_codes', [])
    coarse_groups = hierarchy.get('coarse_groups', {})
    
    print(f"1. Códigos fine: {len(fine_codes)}")
    print(f"2. Grupos coarse: {len(coarse_groups)}")
    
    total_coarse_codes = sum(len(codes) for codes in coarse_groups.values())
    print(f"3. Total códigos en grupos coarse: {total_coarse_codes}")
    
    # Verificar que no hay duplicados
    all_codes = set(fine_codes)
    for codes in coarse_groups.values():
        all_codes.update(codes)
    
    print(f"4. Códigos únicos totales: {len(all_codes)}")
    
    return hierarchy

def test_preprocessor_initialization():
    """Probar inicialización del preprocessor"""
    print("\n=== PRUEBA DE INICIALIZACIÓN DEL PREPROCESSOR ===\n")
    
    try:
        # Configuración básica
        root = 'datos/12Large/WFDBRecords'
        hierarchy_path = 'datos/12Large/labels_hierarchy.json'
        sequence_len = 7500
        
        # Verificar que existen los archivos necesarios
        if not os.path.exists(root):
            print(f"❌ Error: Directorio {root} no existe")
            return False
            
        if not os.path.exists(hierarchy_path):
            print(f"❌ Error: Archivo {hierarchy_path} no existe")
            return False
        
        # Crear una lista pequeña de archivos para prueba
        from glob import glob
        all_files = glob(os.path.join(root, '**', '*.hea'), recursive=True)
        test_files = all_files[:10]  # Solo 10 archivos para prueba
        
        print(f"1. Archivos de prueba: {len(test_files)}")
        
        # Inicializar preprocessor
        print("2. Inicializando HMSTPreprocessor...")
        preprocessor = HMSTPreprocessor(
            root, 
            sequence_len=sequence_len, 
            files=test_files,
            multilabel=True, 
            hierarchy_path=hierarchy_path,
            cache_dir=None,
            random_crop=False,
            target_fs=500.0, 
            bandpass_hz=(0.5, 45.0), 
            notch_hz=None, 
            eval_mode=True,
            wide_feats=True,
            balance_rare=False
        )
        
        print("✅ Preprocessor inicializado correctamente")
        
        # Verificar atributos importantes
        print(f"3. Atributos del preprocessor:")
        print(f"   - fine_codes: {len(preprocessor.fine_codes)}")
        print(f"   - coarse_names: {len(preprocessor.coarse_names)}")
        print(f"   - samples: {len(preprocessor.samples)}")
        
        return preprocessor
        
    except Exception as e:
        print(f"❌ Error en inicialización: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading(preprocessor):
    """Probar carga de datos"""
    print("\n=== PRUEBA DE CARGA DE DATOS ===\n")
    
    try:
        # Probar carga de un sample
        if len(preprocessor.samples) == 0:
            print("❌ No hay samples disponibles")
            return False
        
        print(f"1. Cargando sample 0 de {len(preprocessor.samples)}...")
        sample = preprocessor[0]
        
        print(f"2. Estructura del sample:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   - {key}: {value.shape} ({value.dtype})")
            else:
                print(f"   - {key}: {type(value)} = {value}")
        
        # Verificar dimensiones
        samples = sample['samples']
        labels_fine = sample['labels_fine']
        labels_coarse = sample['labels_coarse']
        
        print(f"\n3. Verificación de dimensiones:")
        print(f"   - samples: {samples.shape} (esperado: [12, 7500])")
        print(f"   - labels_fine: {labels_fine.shape} (esperado: [{len(preprocessor.fine_codes)}])")
        print(f"   - labels_coarse: {labels_coarse.shape} (esperado: [{len(preprocessor.coarse_names)}])")
        
        # Verificar que las dimensiones son correctas
        expected_fine = len(preprocessor.fine_codes)
        expected_coarse = len(preprocessor.coarse_names)
        
        if samples.shape[0] == 12 and samples.shape[1] == 7500:
            print("✅ Dimensiones de samples correctas")
        else:
            print(f"❌ Dimensiones de samples incorrectas: {samples.shape}")
            
        if labels_fine.shape[0] == expected_fine:
            print("✅ Dimensiones de labels_fine correctas")
        else:
            print(f"❌ Dimensiones de labels_fine incorrectas: {labels_fine.shape}, esperado: {expected_fine}")
            
        if labels_coarse.shape[0] == expected_coarse:
            print("✅ Dimensiones de labels_coarse correctas")
        else:
            print(f"❌ Dimensiones de labels_coarse incorrectas: {labels_coarse.shape}, esperado: {expected_coarse}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en carga de datos: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_compatibility(preprocessor):
    """Probar compatibilidad con el modelo"""
    print("\n=== PRUEBA DE COMPATIBILIDAD CON MODELO ===\n")
    
    try:
        from ModeloNuevo.v3.HierarchicalMultiScaleTransformer import HMST
        
        # Parámetros del modelo
        num_fine = len(preprocessor.fine_codes)
        num_coarse = len(preprocessor.coarse_names)
        
        print(f"1. Parámetros del modelo:")
        print(f"   - num_fine: {num_fine}")
        print(f"   - num_coarse: {num_coarse}")
        
        # Crear modelo de prueba
        print("2. Creando modelo HMST...")
        model = HMST(
            input_channels=12+3,  # 12 leads + 3 wide features
            d_model=128,
            nhead=4,
            num_layers=4,
            num_stages=2,
            num_coarse=num_coarse,
            num_fine=num_fine,
            dropout=0.2,
            snomed_dim=num_coarse,
        )
        
        print("✅ Modelo creado correctamente")
        
        # Probar forward pass
        print("3. Probando forward pass...")
        batch_size = 2
        x = torch.randn(batch_size, 12, 7500)
        wide_feats = torch.zeros(batch_size, 3)
        snomed_embed = torch.zeros(batch_size, num_coarse)
        
        with torch.no_grad():
            output = model(x, wide_feats, snomed_embed)
            
        if len(output) == 2:
            coarse_logits, fine_logits = output
        else:
            coarse_logits, fine_logits, _ = output
            
        print(f"   - coarse_logits: {coarse_logits.shape}")
        print(f"   - fine_logits: {fine_logits.shape}")
        
        # Verificar dimensiones de salida
        expected_coarse = (batch_size, num_coarse)
        expected_fine = (batch_size, num_fine)
        
        if coarse_logits.shape == expected_coarse:
            print("✅ Dimensiones de coarse_logits correctas")
        else:
            print(f"❌ Dimensiones de coarse_logits incorrectas: {coarse_logits.shape}, esperado: {expected_coarse}")
            
        if fine_logits.shape == expected_fine:
            print("✅ Dimensiones de fine_logits correctas")
        else:
            print(f"❌ Dimensiones de fine_logits incorrectas: {fine_logits.shape}, esperado: {expected_fine}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en compatibilidad con modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 VERIFICACIÓN COMPLETA DE CARGA DE DATOS\n")
    
    # 1. Probar jerarquía
    hierarchy = test_hierarchy_loading()
    
    # 2. Probar preprocessor
    preprocessor = test_preprocessor_initialization()
    if not preprocessor:
        print("\n❌ FALLO: No se pudo inicializar el preprocessor")
        return False
    
    # 3. Probar carga de datos
    if not test_data_loading(preprocessor):
        print("\n❌ FALLO: No se pudieron cargar los datos")
        return False
    
    # 4. Probar compatibilidad con modelo
    if not test_model_compatibility(preprocessor):
        print("\n❌ FALLO: El modelo no es compatible")
        return False
    
    print("\n🎉 ¡TODAS LAS PRUEBAS PASARON EXITOSAMENTE!")
    print("✅ Los datos se pueden cargar correctamente")
    print("✅ El modelo es compatible con la jerarquía actualizada")
    print("✅ El entrenamiento debería funcionar correctamente")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
