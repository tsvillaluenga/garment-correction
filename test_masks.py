#!/usr/bin/env python3
"""
Script simple para probar la carga de máscaras con transparencia.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from data import load_mask

def test_mask_loading():
    """Probar la carga de máscaras."""
    print("🔍 PROBANDO CARGA DE MÁSCARAS CON TRANSPARENCIA")
    print("=" * 60)
    
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
                        if len(test_masks) >= 3:  # Probar solo 3 máscaras
                            break
        if len(test_masks) >= 3:
            break
    
    if not test_masks:
        print("❌ No se encontraron máscaras para probar")
        return
    
    print(f"📊 Probando {len(test_masks)} máscaras...")
    
    total_ratio = 0
    for i, mask_path in enumerate(test_masks):
        try:
            mask = load_mask(mask_path, size=(256, 256))
            ratio = mask.mean()
            total_ratio += ratio
            
            print(f"✅ Máscara {i+1}: {mask_path.name}")
            print(f"   - Proporción blanca: {ratio:.1%}")
            print(f"   - Forma: {mask.shape}")
            
            if ratio > 0.95:
                print("   ⚠️  Máscara muy blanca (posible problema)")
            elif ratio < 0.05:
                print("   ⚠️  Máscara muy negra (posible problema)")
            else:
                print("   ✅ Máscara con proporción normal")
                
        except Exception as e:
            print(f"❌ Error en {mask_path.name}: {e}")
    
    avg_ratio = total_ratio / len(test_masks)
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   - Promedio de proporción blanca: {avg_ratio:.1%}")
    
    if avg_ratio < 0.9:
        print("✅ ¡Corrección exitosa! Las máscaras tienen proporciones normales.")
        print("💡 Ahora puedes entrenar normalmente:")
        print("   python -m src.cli.train_model2 --config configs/model2.yaml")
    else:
        print("⚠️  Aún hay muchas máscaras muy blancas.")
        print("💡 Puede ser que las máscaras realmente sean completamente blancas.")

if __name__ == "__main__":
    test_mask_loading()
