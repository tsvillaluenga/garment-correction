#!/usr/bin/env python3
"""
Script para filtrar elementos con máscaras problemáticas.
"""
import cv2
import numpy as np
from pathlib import Path
import shutil

def filter_problematic_items(data_root: str, max_white_ratio: float = 0.9, min_white_ratio: float = 0.05):
    """Filtrar elementos con máscaras problemáticas."""
    data_path = Path(data_root)
    filtered_dir = Path(f"{data_root}_filtered")
    
    # Crear directorio filtrado
    filtered_dir.mkdir(exist_ok=True)
    
    stats = {
        'total': 0,
        'kept': 0,
        'filtered_high': 0,
        'filtered_low': 0
    }
    
    for split in ["train", "val"]:
        split_dir = data_path / split
        filtered_split_dir = filtered_dir / split
        
        if not split_dir.exists():
            continue
            
        filtered_split_dir.mkdir(exist_ok=True)
        
        for item_dir in split_dir.iterdir():
            if not item_dir.is_dir():
                continue
                
            stats['total'] += 1
            
            # Verificar máscaras
            mask_still_path = item_dir / "mask_still.png"
            mask_on_model_path = item_dir / "mask_on_model.png"
            
            if not (mask_still_path.exists() and mask_on_model_path.exists()):
                continue
            
            # Analizar máscaras
            mask_still = cv2.imread(str(mask_still_path), cv2.IMREAD_GRAYSCALE)
            mask_on_model = cv2.imread(str(mask_on_model_path), cv2.IMREAD_GRAYSCALE)
            
            if mask_still is None or mask_on_model is None:
                continue
            
            still_ratio = np.sum(mask_still > 127) / mask_still.size
            on_model_ratio = np.sum(mask_on_model > 127) / mask_on_model.size
            
            # Verificar si las máscaras son problemáticas
            still_problematic = still_ratio > max_white_ratio or still_ratio < min_white_ratio
            on_model_problematic = on_model_ratio > max_white_ratio or on_model_ratio < min_white_ratio
            
            if still_problematic or on_model_problematic:
                if still_ratio > max_white_ratio or on_model_ratio > max_white_ratio:
                    stats['filtered_high'] += 1
                    reason = "máscara demasiado blanca"
                else:
                    stats['filtered_low'] += 1
                    reason = "máscara demasiado negra"
                
                print(f"🚫 Filtrando {item_dir.name}: {reason} (still: {still_ratio:.1%}, on_model: {on_model_ratio:.1%})")
            else:
                # Copiar elemento completo
                dest_dir = filtered_split_dir / item_dir.name
                shutil.copytree(item_dir, dest_dir)
                stats['kept'] += 1
    
    return stats

def main():
    print("🔧 FILTRADO DE DATOS PROBLEMÁTICOS")
    print("=" * 50)
    
    # Filtrar elementos problemáticos
    stats = filter_problematic_items("dataset", max_white_ratio=0.9, min_white_ratio=0.05)
    
    print(f"\n📊 RESULTADOS DEL FILTRADO:")
    print(f"   - Total elementos: {stats['total']}")
    print(f"   - Elementos conservados: {stats['kept']}")
    print(f"   - Filtrados (muy blancos): {stats['filtered_high']}")
    print(f"   - Filtrados (muy negros): {stats['filtered_low']}")
    print(f"   - Porcentaje conservado: {stats['kept']/stats['total']*100:.1f}%")
    
    print(f"\n💡 PRÓXIMOS PASOS:")
    print("1. Usar dataset_filtered para entrenar")
    print("2. Revisar elementos filtrados manualmente")
    print("3. Corregir máscaras problemáticas si es necesario")
    
    print(f"\n🚀 COMANDO PARA ENTRENAR CON DATOS FILTRADOS:")
    print("python -m src.cli.train_model2 --config configs/model2.yaml --data_root dataset_filtered")

if __name__ == "__main__":
    main()
