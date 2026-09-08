import numpy as np
import pandas as pd


def _valor_mas_frecuente(serie):
    serie = serie.dropna()
    return serie.mode().iloc[0] if not serie.empty else None


# EXTENSIÓN PROPIA — no viene de ninguno de los 4 ejemplos de referencia.
# Ninguno de los 4 ejemplos clasifica productos por importancia (ABC) ni
# decide por producto si conviene usar ML o un baseline simple; esa
# decisión (usar_ml) es la que hace que el sistema no dependa ciegamente
# de XGBoost para todo el catálogo (ver conversación de metodología).
def calcular_comportamiento_producto(df, df_mensual):
    """
    Clasifica cada producto por importancia (ABC, según % acumulado de
    demanda) y por variabilidad (coeficiente de variación), decide si
    conviene pronosticarlo con XGBoost o con Media Móvil (usar_ml), y le
    agrega etiquetas descriptivas (categoría, nombre) para el dashboard —
    esto último tampoco viene de los ejemplos, es una mejora de usabilidad
    propia (los productos del CSV original solo tienen un ID numérico).
    """
    producto_comportamiento = (
        df_mensual.groupby("producto_id")["demanda"]
        .agg(demanda_total="sum", media="mean", desviacion="std")
        .reset_index()
    )
    producto_comportamiento["cv"] = (
        producto_comportamiento["desviacion"] / producto_comportamiento["media"]
    ).replace([np.inf, -np.inf], np.nan)

    def clasificar_variabilidad(cv):
        if pd.isna(cv):
            return "Sin datos"
        if cv < 0.5:
            return "Baja"
        elif cv < 1:
            return "Media"
        else:
            return "Alta"

    producto_comportamiento["variabilidad"] = producto_comportamiento["cv"].apply(clasificar_variabilidad)

    producto_comportamiento = producto_comportamiento.sort_values("demanda_total", ascending=False).reset_index(drop=True)
    total_demanda = producto_comportamiento["demanda_total"].sum()
    if total_demanda > 0:
        producto_comportamiento["demanda_acumulada_pct"] = producto_comportamiento["demanda_total"].cumsum() / total_demanda
    else:
        producto_comportamiento["demanda_acumulada_pct"] = 0

    def clasificar_abc(pct):
        if pct <= 0.80:
            return "A"
        elif pct <= 0.95:
            return "B"
        else:
            return "C"

    producto_comportamiento["categoria_abc"] = producto_comportamiento["demanda_acumulada_pct"].apply(clasificar_abc)
    producto_comportamiento["usar_ml"] = (
        producto_comportamiento["categoria_abc"].isin(["A", "B"])
        | (producto_comportamiento["variabilidad"] == "Alta")
    )

    # Etiquetas descriptivas del producto (categoría y, si el CSV la
    # trae, nombre). Los productos solo tienen un ID numérico — sin
    # esto el dashboard sería una tabla de números sin contexto. Se usa
    # el valor MÁS FRECUENTE por producto en los datos crudos (moda),
    # no el "primero": en un inventario real la categoría de un
    # producto no cambia mes a mes, pero si el CSV tiene alguna fila
    # inconsistente, la moda es más representativa que quedarse con
    # cualquier fila al azar.
    columnas_etiqueta = [c for c in ["categoria", "nombre_producto"] if c in df.columns]
    if columnas_etiqueta:
        etiquetas_producto = (
            df.groupby("producto_id")[columnas_etiqueta]
            .agg(_valor_mas_frecuente)
            .reset_index()
        )
        producto_comportamiento = producto_comportamiento.merge(
            etiquetas_producto, on="producto_id", how="left"
        )
    for col in ["categoria", "nombre_producto"]:
        if col not in producto_comportamiento.columns:
            producto_comportamiento[col] = None
        else:
            producto_comportamiento[col] = producto_comportamiento[col].where(
                producto_comportamiento[col].notna(), None
            )

    return producto_comportamiento
