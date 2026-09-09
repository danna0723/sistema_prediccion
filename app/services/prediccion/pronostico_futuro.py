import numpy as np
import pandas as pd

from app.services.prediccion.utils import nan_a_none

NOMBRES_MES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

HORIZONTE_MESES = 3


# ============================================================
# PRONÓSTICO REAL DE LOS PRÓXIMOS 3 MESES (recursivo)
# ============================================================
# EXTENSIÓN PROPIA — no viene de ninguno de los 4 ejemplos de
# referencia del proyecto (ver conversación de metodología). Ningún
# ejemplo pronostica varios pasos hacia adelante de forma recursiva;
# esta sección extiende el enfoque de un solo paso (Ejemplo 1/3) a un
# horizonte de 3 meses.
#
# A diferencia de la evaluación de backtest (que mide el modelo contra
# meses que YA pasaron), esta función pronostica los 3 meses SIGUIENTES
# al último mes disponible en el CSV. Como esos meses todavía no
# ocurrieron, no hay demanda real con la cual comparar — por eso no se
# calcula ningún error acá.
#
# Es un pronóstico RECURSIVO (multi-step): la predicción del mes 1 se
# usa como si fuera dato real para calcular los lags del mes 2, la del
# mes 2 para el mes 3, etc. Es el enfoque estándar para extender un
# modelo de un paso a varios pasos sin tener que entrenar un modelo
# distinto por horizonte (con ~12 meses de historia, partir los datos
# en más subconjuntos los haría aún más chicos). El costo es que el
# error se puede ir acumulando de un mes a otro.
def generar_pronostico_futuro(df_mensual, producto_comportamiento, model, feature_cols):
    """
    Devuelve (pronostico_futuro_df, pronostico_pivot,
    meses_pronosticados, meses_pronosticados_legibles).
    """
    proximo_mes = df_mensual["fecha"].max() + pd.DateOffset(months=1)
    meses_pronosticados = [
        (proximo_mes + pd.DateOffset(months=h)).strftime("%Y-%m")
        for h in range(HORIZONTE_MESES)
    ]
    meses_pronosticados_legibles = [
        f"{NOMBRES_MES_ES[(proximo_mes + pd.DateOffset(months=h)).month]} "
        f"{(proximo_mes + pd.DateOffset(months=h)).year}"
        for h in range(HORIZONTE_MESES)
    ]

    # Preparar el estado de cada producto UNA sola vez (no en cada
    # horizonte): historial de demandas, última fila de df_mensual y su
    # fila de segmentación. Se arma con groupby/set_index en vez de
    # filtrar df_mensual/producto_comportamiento con una máscara booleana
    # por cada producto (esto era O(n²) — con 856 productos, cientos de
    # miles de comparaciones de más solo para armar el estado inicial).
    producto_comportamiento_idx = producto_comportamiento.set_index("producto_id")
    estado_por_producto = {}
    for producto, historial_producto in df_mensual.sort_values("fecha").groupby("producto_id"):
        demandas = list(historial_producto["demanda"].values)
        if len(demandas) < 3:
            # No hay suficiente historial para calcular lag_1..lag_3
            continue
        estado_por_producto[producto] = {
            "demandas": demandas,
            "ultima_fila": historial_producto.iloc[-1],
            "segmento": producto_comportamiento_idx.loc[producto],
        }

    productos_validos = sorted(estado_por_producto.keys())

    pronostico_futuro = []

    # Recursivo, pero por HORIZONTE en vez de por producto: en cada paso
    # se arma un lote con la fila de features de los ~800+ productos y
    # se llama a model.predict() UNA vez para todo el lote, en lugar de
    # una vez por producto (2568 llamadas individuales a XGBoost, cada
    # una con su propio overhead de por medio, es la parte más lenta de
    # todo el pipeline — pasar a predicción por lotes lo resuelve sin
    # cambiar ningún cálculo, cada producto se sigue prediciendo con
    # exactamente los mismos datos que antes, solo que juntos).
    for h in range(HORIZONTE_MESES):
        mes_objetivo = proximo_mes + pd.DateOffset(months=h)

        filas_features = []
        metadatos = []

        for producto in productos_validos:
            estado = estado_por_producto[producto]
            demandas = estado["demandas"]
            ultima_fila = estado["ultima_fila"]

            lag_1 = demandas[-1]
            lag_2 = demandas[-2]
            lag_3 = demandas[-3]
            lag_6 = demandas[-6] if len(demandas) >= 6 else 0

            media_movil_3 = np.mean(demandas[-3:])
            media_movil_6 = np.mean(demandas[-6:]) if len(demandas) >= 6 else np.mean(demandas)
            desviacion_movil_3 = np.std(demandas[-3:])

            fila_features = {
                "año": mes_objetivo.year,
                "mes_sin": np.sin(2 * np.pi * mes_objetivo.month / 12),
                "mes_cos": np.cos(2 * np.pi * mes_objetivo.month / 12),
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_6": lag_6,
                "media_movil_3": media_movil_3,
                "media_movil_6": media_movil_6,
                "desviacion_movil_3": desviacion_movil_3,
                "media_producto": ultima_fila.get("media_producto", 0),
                "desviacion_producto": ultima_fila.get("desviacion_producto", 0),
            }

            if "precio" in feature_cols:
                fila_features["precio"] = ultima_fila.get("precio", 0)

            if "inventario" in feature_cols:
                fila_features["inventario"] = ultima_fila.get("inventario", 0)
                if "inventario_lag_1" in feature_cols:
                    fila_features["inventario_lag_1"] = ultima_fila.get("inventario", 0)

            filas_features.append(fila_features)
            metadatos.append((producto, media_movil_3))

        X_futuro_lote = pd.DataFrame(filas_features)[feature_cols]
        preds_xgb_lote = np.maximum(np.expm1(model.predict(X_futuro_lote)), 0)

        for (producto, media_movil_3), pred_xgb_futuro in zip(metadatos, preds_xgb_lote):
            estado = estado_por_producto[producto]
            segmento = estado["segmento"]

            pred_xgb_futuro = float(pred_xgb_futuro)
            pred_baseline_futuro = float(max(media_movil_3, 0))

            usar_ml_producto = bool(segmento["usar_ml"])
            if media_movil_3 > 0 and pred_xgb_futuro < media_movil_3 * 0.5:
                usar_ml_producto = False
            pred_final_futuro = pred_xgb_futuro if usar_ml_producto else pred_baseline_futuro

            pronostico_futuro.append({
                "producto_id": producto,
                "nombre_producto": segmento.get("nombre_producto"),
                "categoria": segmento.get("categoria"),
                "mes_pronosticado": mes_objetivo.strftime("%Y-%m"),
                "horizonte": h + 1,
                "prediccion_xgboost": round(pred_xgb_futuro, 2),
                "prediccion_baseline": round(pred_baseline_futuro, 2),
                "prediccion_final": round(pred_final_futuro, 2),
                "metodo_usado": "XGBoost" if usar_ml_producto else "Media móvil",
            })

            # La predicción de este mes alimenta los lags del siguiente
            estado["demandas"].append(pred_final_futuro)

    pronostico_futuro_df = pd.DataFrame(pronostico_futuro)
    # Ver utils.nan_a_none: reconstruir el DataFrame desde una lista de
    # diccionarios puede convertir a NaN los None de nombre_producto/categoria.
    pronostico_futuro_df = nan_a_none(pronostico_futuro_df, ["nombre_producto", "categoria"])

    pronostico_pivot = []
    if not pronostico_futuro_df.empty:
        for producto, grupo in pronostico_futuro_df.sort_values("horizonte").groupby("producto_id"):
            valores = grupo["prediccion_final"].tolist()
            pronostico_pivot.append({
                "producto_id": producto,
                "nombre_producto": grupo["nombre_producto"].iloc[0],
                "categoria": grupo["categoria"].iloc[0],
                # Redondeado a unidades enteras para la vista del usuario
                # final: fracciones de unidad ("45.63") no aportan nada
                # al que tiene que decidir cuánto pedir.
                "valores": [round(v) for v in valores],
                "total": round(sum(valores)),
            })
        pronostico_pivot.sort(key=lambda fila: fila["total"], reverse=True)

    return pronostico_futuro_df, pronostico_pivot, meses_pronosticados, meses_pronosticados_legibles
