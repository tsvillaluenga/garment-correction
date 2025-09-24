#!/usr/bin/env python3
"""
Script para inspeccionar máscaras problemáticas.
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def inspect_mask(mask_path: str):
    """Inspeccionar una máscara específica."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"❌ No se pudo cargar: {mask_path}")
        return
    
    # Calcular estadísticas
    total_pixels = mask.size
    white_pixels = np.sum(mask > 127)
    black_pixels = total_pixels - white_pixels
    
    white_ratio = white_pixels / total_pixels
    black_ratio = black_pixels / total_pixels
    
    print(f"📊 Análisis de: {mask_path}")
    print(f"   - Tamaño: {mask.shape}")
    print(f"   - Píxeles blancos (prenda): {white_pixels:,} ({white_ratio:.1%})")
    print(f"   - Píxeles negros (fondo): {black_pixels:,} ({black_ratio:.1%})")
    
    # Clasificar el tipo de máscara
    if white_ratio > 0.95:
        print("🚨 PROBLEMA: Máscara casi completamente blanca")
    elif white_ratio < 0.05:
        print("🚨 PROBLEMA: Máscara casi completamente negra")
    elif 0.1 <= white_ratio <= 0.8:
        print("✅ BIEN: Máscara con proporción razonable")
    else:
        print("⚠️  ADVERTENCIA: Proporción inusual pero posible")
    
    return white_ratio

def find_problematic_masks(data_root: str, threshold: float = 0.95):
    """Encontrar todas las máscaras problemáticas."""
    data_path = Path(data_root)
    problematic = []
    
    for split in ["train", "val"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
            
        for item_dir in split_dir.iterdir():
            if not item_dir.is_dir():
                continue
                
            mask_path = item_dir / "mask_still.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    white_ratio = np.sum(mask > 127) / mask.size
                    if white_ratio >= threshold:
                        problematic.append((str(mask_path), white_ratio))
    
    return problematic

def main():
    print("🔍 INSPECCIÓN DE MÁSCARAS PROBLEMÁTICAS")
    print("=" * 50)
    
    # Buscar máscaras problemáticas
    problematic = find_problematic_masks("dataset", threshold=0.95)
    
    print(f"📊 Encontradas {len(problematic)} máscaras problemáticas:")
    
    for mask_path, ratio in problematic[:10]:  # Mostrar solo las primeras 10
        print(f"\n🚨 {mask_path}")
        print(f"   - Proporción blanca: {ratio:.1%}")
        
        # Mostrar imagen original también
        img_path = mask_path.replace("mask_still.png", "still.jpg")
        if Path(img_path).exists():
            print(f"   - Imagen original: {img_path}")
    
    if len(problematic) > 10:
        print(f"\n... y {len(problematic) - 10} más")
    
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   - Total problemáticas: {len(problematic)}")
    print(f"   - Porcentaje del dataset: {len(problematic)/len(problematic)*100:.1f}%")
    
    print(f"\n💡 SOLUCIONES RECOMENDADAS:")
    print("1. Revisar manualmente algunas máscaras problemáticas")
    print("2. Corregir las máscaras incorrectas")
    print("3. Filtrar elementos con máscaras problemáticas")
    print("4. Ajustar el algoritmo de generación de máscaras")

if __name__ == "__main__":
    main()
