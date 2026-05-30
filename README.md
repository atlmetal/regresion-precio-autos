# Regresión: Predecir el precio de autos usados

Solución de los **ejercicios del Módulo 3 (Regresión)** del *Curso de Machine
Learning para Ingeniería de Software*. Se construye, paso a paso, un modelo de
regresión lineal para predecir el precio (`msrp`) de autos usados, se evalúa con
**RMSE** y se mejora con **feature engineering** y **regularización (Ridge)**.

📓 **Notebook con la solución completa:**
[`notebooks/solucion-regresion-precio-autos.ipynb`](notebooks/solucion-regresion-precio-autos.ipynb)

## Ejercicios resueltos

1. **Preparación de datos** — carga, tratamiento de nulos, transformación log del
   target y división train/val/test (60/20/20).
2. **Regresión lineal** — modelo con 2 features, RMSE en validación, ampliación a
   5 features e interpretación de los pesos.
3. **Feature engineering** — variable `antiguedad` y One-Hot Encoding de la marca.
4. **Regularización** — barrido de `alpha` con Ridge, gráfica alpha vs RMSE,
   modelo final y RMSE en test.
5. **Comparación final** — tabla resumen de todos los modelos.

## Resultados (RMSE sobre `log1p(precio)`)

| Modelo                      | Features          | RMSE val |
|-----------------------------|-------------------|:--------:|
| Baseline (media)            | —                 |  1.1296  |
| Lineal (2 features)         | year, hp          |  0.5332  |
| Lineal (todas + OHE marca)  | numéricas + marca |  0.4479  |
| Ridge (mejor alpha = 1)     | numéricas + marca |  0.4545  |

**Modelo final** (Ridge `alpha = 1`, entrenado en train + val) → **RMSE en test = 0.4523**.

> Hallazgo: el mayor avance proviene del One-Hot Encoding de la marca, no del
> algoritmo. Ridge no mejora a la regresión lineal simple porque la matriz de
> features no es inestable; la regularización ayuda más ante colinealidad fuerte
> o pocos datos.

## Cómo ejecutarlo

Requiere [`uv`](https://docs.astral.sh/uv/).

```bash
# 1. Crear el entorno e instalar dependencias
uv sync --python 3.13

# 2. Abrir el notebook
uv run jupyter notebook notebooks/solucion-regresion-precio-autos.ipynb
```

El dataset (`notebooks/data.csv`) ya está incluido en el repositorio.

## Tecnologías

NumPy · pandas · scikit-learn · Matplotlib · seaborn · Jupyter
