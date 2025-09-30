# Cambios implementados: Análisis explícito de colores para Modelo 1

## Resumen

El Modelo 1 ahora analiza **explícitamente** los colores dominantes en las regiones enmascaradas de `still` y `on_model`, y aprende a alinear sus distribuciones de color.

## Archivos modificados

### 1. **`src/color_analysis.py`** (NUEVO)
Módulo completo para análisis explícito de colores:

#### Funcionalidades principales:
- **`extract_masked_pixels()`**: Extrae píxeles de la región enmascarada
- **`filter_shadows_highlights()`**: Filtra sombras y brillos (a menos que sean el color base)
- **`detect_garment_type()`**: Detecta si la prenda es oscura o clara
- **`extract_dominant_colors()`**: Usa K-means en espacio LAB para extraer colores dominantes
- **`compute_color_distribution()`**: Calcula histogramas RGB normalizados
- **`analyze_image()`**: Análisis completo de una imagen enmascarada
- **`compare_colors()`**: Compara distribuciones de color entre still y on-model

#### Parámetros clave:
```python
ColorAnalyzer(
    n_colors=5,                    # Número de colores dominantes
    shadow_percentile=10.0,        # Percentil para considerar sombras
    highlight_percentile=90.0,     # Percentil para considerar brillos
    min_pixels=100                 # Mínimo de píxeles para análisis
)
```

#### Salida del análisis:
```python
{
    'dominant_colors': np.ndarray,    # (K, 3) colores dominantes en RGB
    'color_weights': np.ndarray,      # (K,) proporción de cada color
    'n_colors': int,                  # Número de colores detectados
    'mean_color': np.ndarray,         # (3,) color promedio ponderado
    'histogram': np.ndarray,          # (96,) histograma RGB concatenado
    'is_dark': bool,                  # ¿Prenda oscura?
    'is_light': bool,                 # ¿Prenda clara?
    'n_pixels': int,                  # Píxeles totales
    'n_filtered_pixels': int          # Píxeles después de filtrar
}
```

---

### 2. **`src/losses_metrics.py`**
Añadida nueva pérdida `ColorAlignmentLoss`:

#### Componentes:
```python
ColorAlignmentLoss(
    w_mean=1.0,        # Peso para diferencia de color promedio
    w_hist=0.5,        # Peso para correlación de histogramas
    w_dominant=0.5     # Peso para diferencia de colores dominantes
)
```

#### Cálculo:
1. **Mean color loss**: MSE entre colores promedios
2. **Histogram loss**: 1 - cosine_similarity entre histogramas
3. **Dominant color loss**: MSE entre los 3 colores dominantes principales

#### Integración en `CombinedRecolorLoss`:
```python
CombinedRecolorLoss(
    w_l1=1.0,
    w_de=0.5,
    w_perc=0.01,
    w_gan=0.0,
    w_color=0.3      # NUEVO: Color alignment loss
)
```

---

### 3. **`configs/model1.yaml`**
Añadido peso para color alignment:

```yaml
loss_weights:
  w_l1: 1.0
  w_de: 0.5
  w_perc: 0.01
  w_gan: 0.0
  w_color: 0.3  # NUEVO: Color alignment loss
```

---

### 4. **`src/cli/train_model1.py`**
Modificado para pasar `still_ref` y `mask_still` a la función de pérdida:

#### Cambios en training loop:
```python
# ANTES:
loss_dict = criterion(pred, on_model_target, mask_on, fake_logits)

# AHORA:
loss_dict = criterion(pred, on_model_target, mask_on, fake_logits, still_ref, mask_still)
```

#### Cambios en validation loop:
```python
# ANTES:
loss_dict = criterion(pred, on_model_target, mask_on, fake_logits)

# AHORA:
loss_dict = criterion(pred, on_model_target, mask_on, fake_logits, still_ref, mask_still)
```

---

### 5. **`requirements.txt`**
Añadida dependencia:

```txt
scikit-learn>=1.3.0  # Para K-means clustering
```

---

## Cómo funciona el nuevo sistema

### Durante el entrenamiento:

1. **Análisis de `still`**:
   - Extrae píxeles en `mask_still`
   - Detecta si la prenda es oscura/clara
   - Filtra sombras/brillos (si no son el color base)
   - Extrae 5 colores dominantes usando K-means en LAB
   - Calcula color promedio y histograma

2. **Análisis de `on_model_target`**:
   - Mismo proceso con `mask_on_model`
   - Extrae colores dominantes, promedio, histograma

3. **Análisis de `pred` (predicción)**:
   - Mismo proceso con `mask_on_model`

4. **Pérdida de alineación de color**:
   ```python
   L_color = w_mean * MSE(mean_pred, mean_target) +
             w_hist * (1 - cosine_sim(hist_pred, hist_target)) +
             w_dominant * MSE(dominant_pred, dominant_target)
   ```

5. **Pérdida total**:
   ```python
   L_total = w_l1 * L_L1 + 
             w_de * L_ΔE + 
             w_perc * L_perceptual + 
             w_color * L_color
   ```

### Durante la inferencia:

- **NO requiere cambios** en `infer_recolor.py`
- El modelo ya aprendió a alinear colores durante entrenamiento
- La inferencia usa el mismo `forward_infer()` de antes

---

## Ventajas del nuevo enfoque

### 1. **Análisis explícito**
- No depende solo de features aprendidas
- Extrae colores de forma interpretable
- Permite debugging y visualización

### 2. **Filtrado de sombras/brillos**
- Distingue entre color base y variaciones de iluminación
- Mantiene sombras si la prenda es oscura
- Mantiene brillos si la prenda es clara

### 3. **Espacio LAB**
- K-means en LAB → colores más perceptualmente uniformes
- Mejor separación de colores similares
- Más robusto que RGB o HSV

### 4. **Multi-escala**
- Color promedio: captura tono general
- Histograma: captura distribución
- Colores dominantes: captura regiones discretas

### 5. **Robustez**
- Mínimo 100 píxeles para análisis
- Fallback a color promedio si hay pocos píxeles
- Normalización de histogramas

---

## Parámetros ajustables

### En `ColorAnalyzer`:
```python
n_colors = 5              # Más colores → más detalle, más tiempo
shadow_percentile = 10.0  # Más bajo → filtra más sombras
highlight_percentile = 90.0  # Más alto → filtra más brillos
min_pixels = 100          # Más alto → más robusto, menos sensible
```

### En `ColorAlignmentLoss`:
```python
w_mean = 1.0      # Importancia del color promedio
w_hist = 0.5      # Importancia de la distribución
w_dominant = 0.5  # Importancia de colores discretos
```

### En `model1.yaml`:
```yaml
w_color: 0.3  # Peso global de la pérdida de color
              # Más alto → más énfasis en color
              # Más bajo → más énfasis en otras pérdidas
```

---

## Comandos para entrenar

### Instalar dependencias:
```bash
pip install -r requirements.txt
```

### Entrenar Modelo 1 con color analysis:
```bash
python -m src.cli.train_model1 --config configs/model1.yaml
```

### El sistema ahora:
1. Extrae colores dominantes de `still` en `mask_still`
2. Extrae colores dominantes de `on_model` en `mask_on_model`
3. Aprende a transferir la distribución de color de `still` a `on_model`
4. Filtra sombras/brillos automáticamente
5. Compara color promedio, histograma y colores dominantes
6. Optimiza con pérdida compuesta

---

## Métricas de logging

Durante el entrenamiento verás nuevas métricas:

```
Loss: 0.1234 L1: 0.0567 ΔE: 1.2345 Perc: 0.0012 
Color_mean: 0.0234 Color_hist: 0.0456 Color_dom: 0.0123
```

- **Color_mean**: Diferencia de color promedio
- **Color_hist**: Diferencia de distribución
- **Color_dom**: Diferencia de colores dominantes

---

## Próximos pasos

1. **Entrenar el modelo** con los nuevos cambios
2. **Observar métricas** de color alignment
3. **Ajustar pesos** si es necesario
4. **Comparar resultados** con versión anterior

¡El modelo ahora aprende explícitamente a transferir colores! 🎨
