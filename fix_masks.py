#!/usr/bin/env python3
"""
Script para convertir máscaras PNG problemáticas a formato correcto.
"""
import cv2
import numpy as np
from pathlib import Path
import argparse

def fix_mask_format(input_path: str, output_path: str = None):
    """Corregir formato de una máscara PNG."""
    if output_path is None:
        output_path = input_path
    
    # Cargar máscara con canal alfa
    mask = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    
    if mask is None:
        print(f"❌ No se pudo cargar: {input_path}")
        return False
    
    # Procesar según el formato
    if len(mask.shape) == 3:
        if mask.shape[2] == 4:  # RGBA
            # Usar canal alfa como máscara
            alpha = mask[:, :, 3]
            # Convertir a formato binario: transparente=0, no transparente=255
            binary_mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
        elif mask.shape[2] == 3:  # RGB
            # Convertir a escala de grises
            binary_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            print(f"❌ Formato no soportado: {mask.shape}")
            return False
    elif len(mask.shape) == 2:
        # Ya es escala de grises
        binary_mask = mask
    else:
        print(f"❌ Formato no soportado: {mask.shape}")
        return False
    
    # Guardar como PNG sin transparencia
    success = cv2.imwrite(output_path, binary_mask)
    
    if success:
        # Verificar resultado
        ratio = np.sum(binary_mask > 127) / binary_mask.size
        print(f"✅ Corregida: {Path(input_path).name} -> {ratio:.1%} blancos")
        return True
    else:
        print(f"❌ Error guardando: {output_path}")
        return False

def batch_fix_masks(data_root: str, backup: bool = True):
    """Corregir todas las máscaras problemáticas en lote."""
    data_path = Path(data_root)
    backup_dir = Path(f"{data_root}_backup") if backup else None
    
    if backup_dir and backup_dir.exists():
        print(f"⚠️  Backup ya existe: {backup_dir}")
        response = input("¿Continuar? (y/n): ")
        if response.lower() != 'y':
            return
    
    stats = {
        'processed': 0,
        'fixed': 0,
        'errors': 0
    }
    
    for split in ["train", "val"]:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
            
        print(f"\n📁 Procesando {split}...")
        
        for item_dir in split_dir.iterdir():
            if not item_dir.is_dir():
                continue
            
            # Procesar máscaras de still
            mask_still_path = item_dir / "mask_still.png"
            if mask_still_path.exists():
                stats['processed'] += 1
                
                # Crear backup si es necesario
                if backup_dir:
                    backup_item_dir = backup_dir / split / item_dir.name
                    backup_item_dir.mkdir(parents=True, exist_ok=True)
                    backup_mask_path = backup_item_dir / "mask_still.png"
                    if not backup_mask_path.exists():
                        import shutil
                        shutil.copy2(mask_still_path, backup_mask_path)
                
                # Corregir máscara
                if fix_mask_format(str(mask_still_path)):
                    stats['fixed'] += 1
                else:
                    stats['errors'] += 1
            
            # Procesar máscaras de on_model
            mask_on_model_path = item_dir / "mask_on_model.png"
            if mask_on_model_path.exists():
                stats['processed'] += 1
                
                # Crear backup si es necesario
                if backup_dir:
                    backup_item_dir = backup_dir / split / item_dir.name
                    backup_item_dir.mkdir(parents=True, exist_ok=True)
                    backup_mask_path = backup_item_dir / "mask_on_model.png"
                    if not backup_mask_path.exists():
                        import shutil
                        shutil.copy2(mask_on_model_path, backup_mask_path)
                
                # Corregir máscara
                if fix_mask_format(str(mask_on_model_path)):
                    stats['fixed'] += 1
                else:
                    stats['errors'] += 1
    
    print(f"\n📊 RESULTADOS:")
    print(f"   - Máscaras procesadas: {stats['processed']}")
    print(f"   - Máscaras corregidas: {stats['fixed']}")
    print(f"   - Errores: {stats['errors']}")
    
    if backup_dir:
        print(f"   - Backup creado en: {backup_dir}")

def main():
    parser = argparse.ArgumentParser(description="Corregir formato de máscaras PNG")
    parser.add_argument("--data_root", default="dataset", help="Directorio del dataset")
    parser.add_argument("--no_backup", action="store_true", help="No crear backup")
    parser.add_argument("--single", help="Corregir una sola máscara")
    
    args = parser.parse_args()
    
    if args.single:
        # Corregir una sola máscara
        fix_mask_format(args.single)
    else:
        # Corregir todo el dataset
        print("🔧 CORRECCIÓN MASIVA DE MÁSCARAS PNG")
        print("=" * 50)
        
        if not args.no_backup:
            print("💾 Se creará backup automático")
        
        batch_fix_masks(args.data_root, backup=not args.no_backup)
        
        print(f"\n✅ Corrección completada!")
        print(f"💡 Ahora puedes entrenar normalmente con:")
        print(f"   python -m src.cli.train_model2 --config configs/model2.yaml")

if __name__ == "__main__":
    main()
