#!/usr/bin/env python3
"""
Script para probar la corrección de carga de máscaras PNG.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from data import load_mask

def test_mask_loading():
    """Probar la carga de máscaras con diferentes métodos."""
    print("🔍 PROBANDO CARGA DE MÁSCARAS PNG")
    print("=" * 50)
    
    # Buscar algunas máscaras para probar
    test_masks = []
    for split in ["train", "val"]:
        split_dir = Path(f"dataset/{split}")
        if split_dir.exists():
            for item_dir in split_dir.iterdir():
                if item_dir.is_dir():
                    mask_path = item_dir / "mask_still.png"
                    if mask_path.exists():
                        test_masks.append(mask_path)
                        if len(test_masks) >= 5:  # Probar solo 5 máscaras
                            break
        if len(test_masks) >= 5:
            break
    
    if not test_masks:
        print("❌ No se encontraron máscaras para probar")
        return
    
    print(f"📊 Probando {len(test_masks)} máscaras...")
    
    for i, mask_path in enumerate(test_masks):
        print(f"\n🔍 Máscara {i+1}: {mask_path.name}")
        
        # Método anterior (problemático)
        try:
            mask_old = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_old is not None:
                old_ratio = np.sum(mask_old > 127) / mask_old.size
                print(f"   Método anterior: {old_ratio:.1%} blancos")
        except Exception as e:
            print(f"   Método anterior: Error - {e}")
        
        # Método nuevo (corregido)
        try:
            mask_new = load_mask(mask_path, size=(256, 256))
            new_ratio = np.mean(mask_new)
            print(f"   Método nuevo: {new_ratio:.1%} blancos")
            
            # Mostrar información adicional
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_raw is not None:
                print(f"   Formato original: {mask_raw.shape}")
                if len(mask_raw.shape) == 3 and mask_raw.shape[2] == 4:
                    alpha_channel = mask_raw[:, :, 3]
                    alpha_ratio = np.sum(alpha_channel > 0) / alpha_channel.size
                    print(f"   Canal alfa: {alpha_ratio:.1%} no transparente")
                    
        except Exception as e:
            print(f"   Método nuevo: Error - {e}")

def analyze_dataset():
    """Analizar el dataset completo con la corrección."""
    print("\n📊 ANÁLISIS DEL DATASET COMPLETO")
    print("=" * 50)
    
    stats = {
        'total': 0,
        'problematic_old': 0,
        'problematic_new': 0,
        'good_masks': 0
    }
    
    for split in ["train", "val"]:
        split_dir = Path(f"dataset/{split}")
        if not split_dir.exists():
            continue
            
        for item_dir in split_dir.iterdir():
            if not item_dir.is_dir():
                continue
                
            mask_path = item_dir / "mask_still.png"
            if not mask_path.exists():
                continue
                
            stats['total'] += 1
            
            try:
                # Método nuevo
                mask_new = load_mask(mask_path)
                new_ratio = np.mean(mask_new)
                
                if new_ratio > 0.95:
                    stats['problematic_new'] += 1
                elif new_ratio < 0.05:
                    stats['problematic_new'] += 1
                else:
                    stats['good_masks'] += 1
                    
            except Exception as e:
                print(f"Error procesando {mask_path}: {e}")
    
    print(f"📈 ESTADÍSTICAS:")
    print(f"   - Total máscaras: {stats['total']}")
    print(f"   - Máscaras problemáticas (nuevo método): {stats['problematic_new']}")
    print(f"   - Máscaras buenas: {stats['good_masks']}")
    print(f"   - Porcentaje buenas: {stats['good_masks']/stats['total']*100:.1f}%")
    
    if stats['problematic_new'] < stats['total'] * 0.1:  # Menos del 10% problemáticas
        print("✅ ¡Corrección exitosa! Dataset limpio para entrenar.")
    else:
        print("⚠️  Aún hay muchas máscaras problemáticas. Revisar formato.")

def main():
    test_mask_loading()
    analyze_dataset()
    
    print(f"\n💡 PRÓXIMOS PASOS:")
    print("1. Si las estadísticas mejoraron, entrenar normalmente")
    print("2. Si aún hay problemas, revisar formato de las máscaras")
    print("3. Considerar regenerar máscaras con formato correcto")

if __name__ == "__main__":
    main()
