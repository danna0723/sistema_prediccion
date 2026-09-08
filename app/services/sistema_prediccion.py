import os
import numpy as np
import pandas as pd
import xgboost as xgb
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Ejemplo 4 (BoomBikes — Regresión Lineal): RFE + escalado + eliminación
# stepwise por p-valor/VIF (statsmodels), usados en _seleccionar_features_regresion.
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)


# ============================================================
# Ejemplo 4 (BoomBikes — "Linear Regression: Demand Forecasting")
# ============================================================
def _seleccionar_features_regresion(X_train, y_train, max_p_valor=0.05, max_vif=5.0):
    """
    Selección de variables para la Regresión Lineal, con la misma lógica de
    dos pasos usada en el caso de referencia (BoomBikes): primero RFE
    (sklearn) para descartar a la mitad las variables candidatas, y luego
    eliminación hacia atrás (statsmodels OLS) quitando de a una la variable
    con mayor p-valor o mayor VIF, hasta que todas las que quedan sean
    significativas (p < 0.05) y no estén colineales entre sí (VIF < 5).

    Antes de RFE se escalan las variables con MinMaxScaler (igual que en
    el notebook de referencia): RFE con un estimador LinearRegression
    decide qué descartar mirando la magnitud del coeficiente, y sin
    escalar, una variable como "año" (~2021) y otra como "lag_1" (cientos
    de unidades de demanda) no son comparables entre sí. El escalador se
    ajusta solo con datos de entrenamiento para no filtrar información
    del set de prueba.
    """
    # Columnas sin variación (p. ej. "año" cuando el archivo subido solo
    # tiene un año de datos) no aportan nada y rompen el cálculo de VIF
    # (división por cero), así que se descartan antes de empezar.
    candidatas = [c for c in X_train.columns if X_train[c].std() > 0]
    if len(candidatas) < 2 or len(X_train) <= len(candidatas) + 1:
        return candidatas

    try:
        scaler = MinMaxScaler()
        X_train_esc = pd.DataFrame(
            scaler.fit_transform(X_train[candidatas]),
            columns=candidatas, index=X_train.index
        )

        n_rfe = max(2, min(8, len(candidatas) - 1))
        rfe = RFE(LinearRegression(), n_features_to_select=n_rfe)
        rfe.fit(X_train_esc, y_train)
        seleccionadas = [c for c, usar in zip(candidatas, rfe.support_) if usar]

        while len(seleccionadas) > 1:
            X_sm = sm.add_constant(X_train_esc[seleccionadas])
            vif = pd.DataFrame({
                "feature": seleccionadas,
                "VIF": [variance_inflation_factor(X_sm.values, i + 1) for i in range(len(seleccionadas))]
            })

            # La colinealidad se resuelve antes de mirar p-valores: con
            # VIF infinito/NaN la matriz de diseño es singular (colinealidad
            # perfecta) y el ajuste OLS que sigue no es confiable todavía,
            # así que primero se descarta la variable más colineal.
            vif_problema = vif[~np.isfinite(vif["VIF"])]
            if not vif_problema.empty:
                seleccionadas.remove(vif_problema.iloc[0]["feature"])
                continue

            peor_vif_fila = vif.loc[vif["VIF"].idxmax()]
            if peor_vif_fila["VIF"] > max_vif:
                seleccionadas.remove(peor_vif_fila["feature"])
                continue

            modelo_ols = sm.OLS(y_train, X_sm).fit()
            p_valores = modelo_ols.pvalues.drop("const")
            peor_p_nombre = p_valores.idxmax()
            peor_p_valor = p_valores[peor_p_nombre]

            if peor_p_valor > max_p_valor:
                seleccionadas.remove(peor_p_nombre)
            else:
                break

        return seleccionadas
    except Exception:
        # Con pocos datos o columnas colineales el ajuste OLS puede fallar
        # (matriz singular, etc.); si pasa, se sigue con todas las
        # variables candidatas en vez de romper el pronóstico.
        return candidatas


def ejecutar_sistema(ruta_archivo, carpeta_resultados=None):

    if carpeta_resultados is None:
        carpeta_resultados = "resultados"

    os.makedirs(carpeta_resultados, exist_ok=True)

    df = pd.read_csv(ruta_archivo)

    df = df.rename(columns={
        "Sale_Date": "fecha", "Date": "fecha", "date": "fecha", "Fecha": "fecha",
        "Product_ID": "producto_id", "Product": "producto_id", "Producto": "producto_id",
        "Quantity_Sold": "demanda", "Sales": "demanda", "Demand": "demanda", "Demanda": "demanda",
        "Unit_Price": "precio", "Price": "precio", "Price_per_unit": "precio",
        "Inventario_Actual": "inventario", "Inventory": "inventario", "Stock": "inventario",
        "Store": "tienda_id", "Store_ID": "tienda_id",
        "Category": "categoria", "Product_Category": "categoria",
        "Product_Name": "nombre_producto", "Nombre_Producto": "nombre_producto", "Nombre": "nombre_producto",
        "Lead_time_Semana": "lead_time_semanas",
        "Costo_pedido": "costo_pedido",
        "Costo_unitario": "costo_unitario",
        "Porcentaje_costo_mantenimiento": "porcentaje_costo_mantenimiento"
    })

    required_columns = ["fecha", "producto_id", "demanda"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["demanda"] = pd.to_numeric(df["demanda"], errors="coerce")
    df = df.dropna(subset=["fecha", "producto_id", "demanda"])
    df["demanda"] = np.maximum(df["demanda"], 0)
    df = df.sort_values(["producto_id", "fecha"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # Descartar el último mes si está incompleto (ej. el CSV corta
    # el día 1 de un mes nuevo, dejando solo 1 día de ese mes en vez
    # de todo el mes). Si no se descarta, distorsiona la evaluación
    # Y el pronóstico futuro (que usa el último mes como lag_1).
    # ------------------------------------------------------------
    ultimo_periodo_crudo = df["fecha"].max().to_period("M")
    fin_ultimo_periodo_crudo = ultimo_periodo_crudo.end_time
    dias_faltantes = (fin_ultimo_periodo_crudo - df["fecha"].max()).days

    if dias_faltantes > 5:
        filas_antes = len(df)
        df = df[df["fecha"].dt.to_period("M") != ultimo_periodo_crudo].copy()
        print(
            f"Se descartó el mes {ultimo_periodo_crudo} por estar incompleto "
            f"(le faltaban {dias_faltantes} días). Filas removidas: {filas_antes - len(df)}."
        )
        if df.empty:
            raise ValueError("Después de descartar el último mes incompleto no quedan datos.")

    df["mes_periodo"] = df["fecha"].dt.to_period("M").dt.to_timestamp()

    agregaciones = {"demanda": "sum"}
    if "precio" in df.columns:
        agregaciones["precio"] = "mean"
    if "inventario" in df.columns:
        agregaciones["inventario"] = "mean"
    if "costo_unitario" in df.columns:
        agregaciones["costo_unitario"] = "mean"
    if "categoria" in df.columns:
        agregaciones["categoria"] = "first"
    for col in ["lead_time_semanas", "costo_pedido", "porcentaje_costo_mantenimiento"]:
        if col in df.columns:
            agregaciones[col] = "first"

    df_mensual = (
        df.groupby(["producto_id", "mes_periodo"])
        .agg(agregaciones)
        .reset_index()
        .rename(columns={"mes_periodo": "fecha"})
    )

    productos = df_mensual["producto_id"].unique()
    meses_completos = pd.date_range(df_mensual["fecha"].min(), df_mensual["fecha"].max(), freq="MS")

    calendario_producto_mes = (
        pd.MultiIndex.from_product([productos, meses_completos], names=["producto_id", "fecha"])
        .to_frame(index=False)
    )

    df_mensual = calendario_producto_mes.merge(df_mensual, on=["producto_id", "fecha"], how="left")
    df_mensual["demanda"] = df_mensual["demanda"].fillna(0)

    columnas_ffill = [
        c for c in
        ["precio", "costo_unitario", "inventario", "categoria",
         "lead_time_semanas", "costo_pedido", "porcentaje_costo_mantenimiento"]
        if c in df_mensual.columns
    ]
    for col in columnas_ffill:
        df_mensual[col] = df_mensual.groupby("producto_id")[col].transform(lambda x: x.ffill().bfill())

    df_mensual = df_mensual.sort_values(["producto_id", "fecha"]).reset_index(drop=True)

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
    def _valor_mas_frecuente(serie):
        serie = serie.dropna()
        return serie.mode().iloc[0] if not serie.empty else None

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

    df_mensual = df_mensual.merge(
        producto_comportamiento[["producto_id", "categoria_abc", "variabilidad", "usar_ml"]],
        on="producto_id", how="left"
    )

    df_mensual["año"] = df_mensual["fecha"].dt.year
    df_mensual["mes"] = df_mensual["fecha"].dt.month

    # Codificación cíclica en vez de "mes"/"trimestre" crudos: con solo
    # ~12 meses de historia, los primeros 3 meses del dataset siempre
    # quedan fuera del entrenamiento (no alcanzan a tener lag_1..lag_3),
    # así que ese valor de mes nunca aparece como target de entrenamiento.
    # seno/coseno lo representan como un punto continuo entre los meses
    # vecinos que sí se entrenaron, en vez de una categoría nunca vista
    # (que hacía que XGBoost extrapolara mal y colapsara la predicción
    # cerca de 0 para el mes siguiente al último del CSV).
    df_mensual["mes_sin"] = np.sin(2 * np.pi * df_mensual["mes"] / 12)
    df_mensual["mes_cos"] = np.cos(2 * np.pi * df_mensual["mes"] / 12)

    # Ejemplo 1 (SKU Demand Forecasting): mismo principio de feature
    # engineering ("Lag features" + "Rolling statistics") que build_features()
    # en el notebook de referencia, adaptado a periodicidad mensual en vez
    # de semanal (lag_1w..lag_52w / roll_mean_4w,12w -> lag_1..lag_6 /
    # media_movil_3,6).
    df_mensual["lag_1"] = df_mensual.groupby("producto_id")["demanda"].shift(1)
    df_mensual["lag_2"] = df_mensual.groupby("producto_id")["demanda"].shift(2)
    df_mensual["lag_3"] = df_mensual.groupby("producto_id")["demanda"].shift(3)
    df_mensual["lag_6"] = df_mensual.groupby("producto_id")["demanda"].shift(6)

    df_mensual["media_movil_3"] = df_mensual.groupby("producto_id")["demanda"].transform(
        lambda x: x.shift(1).rolling(3).mean()
    )
    df_mensual["media_movil_6"] = df_mensual.groupby("producto_id")["demanda"].transform(
        lambda x: x.shift(1).rolling(6).mean()
    )
    df_mensual["desviacion_movil_3"] = df_mensual.groupby("producto_id")["demanda"].transform(
        lambda x: x.shift(1).rolling(3).std()
    )

    ruta_segmentacion_grafica = os.path.join(carpeta_resultados, "segmentacion_abc_variabilidad.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    producto_comportamiento["categoria_abc"].value_counts().sort_index().plot.bar(ax=axes[0])
    axes[0].set_title("Productos por categoría ABC")
    producto_comportamiento["variabilidad"].value_counts().plot.bar(ax=axes[1])
    axes[1].set_title("Productos por variabilidad")
    plt.tight_layout()
    plt.savefig(ruta_segmentacion_grafica, bbox_inches="tight")
    plt.close()

    # Ejemplo 3 (Kaggle Grupo Bimbo): mismo principio que las agregaciones
    # por producto del notebook de referencia (mean_prod/count_prod) —
    # estadísticas históricas por producto usadas como feature adicional,
    # aquí llamadas media_producto/desviacion_producto.
    stats_cols = ["demanda_total", "media", "desviacion"]
    df_mensual = df_mensual.merge(
        producto_comportamiento[["producto_id"] + stats_cols].rename(
            columns={"media": "media_producto", "desviacion": "desviacion_producto"}
        ),
        on="producto_id", how="left"
    )

    if "inventario" in df_mensual.columns:
        df_mensual["inventario_lag_1"] = df_mensual.groupby("producto_id")["inventario"].shift(1)

    feature_history = ["lag_1", "lag_2", "lag_3", "media_movil_3"]
    df_model = df_mensual.dropna(subset=feature_history).copy()

    if len(df_model) == 0:
        raise ValueError("No hay suficientes datos históricos para entrenar el modelo.")

    feature_cols = [
        "año", "mes_sin", "mes_cos",
        "lag_1", "lag_2", "lag_3", "lag_6",
        "media_movil_3", "media_movil_6", "desviacion_movil_3",
        "media_producto", "desviacion_producto"
    ]
    if "precio" in df_model.columns:
        # Solo el NIVEL de precio (estable, se puede seguir usando el
        # último precio conocido al pronosticar). Se excluyó la
        # variación de precio mes a mes: es un feature muy ruidoso
        # (llegaba a acaparar >65% de la importancia del modelo) del
        # que no se puede conocer el valor futuro real, y forzarlo a 0
        # para el pronóstico hacía que el modelo colapsara.
        feature_cols.append("precio")
    if "inventario" in df_model.columns:
        feature_cols.append("inventario")
        if "inventario_lag_1" in df_model.columns:
            feature_cols.append("inventario_lag_1")

    for col in feature_cols:
        if df_model[col].isna().any():
            df_model[col] = df_model[col].fillna(0)

    meses_ordenados = sorted(df_model["fecha"].unique())
    if len(meses_ordenados) < 4:
        raise ValueError("Se necesitan al menos 4 meses de datos después del procesamiento.")

    # Ejemplo 1 / Ejemplo 3: split de entrenamiento/prueba por fecha (no
    # aleatorio), igual que el corte por semana en ambos notebooks de
    # referencia — necesario en series de tiempo para no entrenar con
    # datos "del futuro" respecto al período de prueba.
    meses_test = meses_ordenados[-3:]
    train_df = df_model[~df_model["fecha"].isin(meses_test)].copy()
    test_df = df_model[df_model["fecha"].isin(meses_test)].copy()

    if len(train_df) == 0:
        raise ValueError("No hay suficientes datos para entrenamiento.")

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["demanda"]
    y_test = test_df["demanda"]

    # Ejemplo 3 (Kaggle Grupo Bimbo): log1p del target antes de entrenar
    # XGBoost (estabiliza varianza/sesgo de una variable de demanda con
    # cola larga); se revierte con expm1 más abajo, al predecir.
    y_train_log = np.log1p(y_train)

    # Ejemplo 4 (BoomBikes): Regresión Lineal entrenada solo con las
    # variables que sobrevivieron RFE + eliminación stepwise (p-valor/VIF).
    features_lr = _seleccionar_features_regresion(X_train, y_train)
    if not features_lr:
        features_lr = list(feature_cols)

    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train[features_lr], y_train)
    pred_lr = modelo_lr.predict(X_test[features_lr])
    pred_lr = np.maximum(pred_lr, 0)

    # Ejemplo 1 (SKU Demand Forecasting): mismos hiperparámetros de
    # XGBoost que el notebook de referencia (max_depth, learning_rate,
    # subsample, colsample_bytree, reg_alpha, reg_lambda, random_state).
    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=500, learning_rate=0.05,
        max_depth=4, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train_log)
    pred_xgb = model.predict(X_test)
    # Ejemplo 3: expm1 (inverso de log1p) + clip a 0 para volver a la
    # escala real de unidades de demanda, igual que el notebook de Bimbo.
    pred_xgb = np.maximum(np.expm1(pred_xgb), 0)

    # Ejemplo 1: baseline de Media Móvil (equivalente a la "4-wk Moving
    # Average" del notebook de referencia, aquí a 3 meses) — se usa tanto
    # de piso de comparación en la tabla de métricas como de método de
    # respaldo real del pronóstico (ver "usar_ml" más abajo).
    pred_baseline_3 = np.maximum(test_df["media_movil_3"].values, 0)

    def calcular_wape(y_real, predicciones):
        denominador = np.sum(np.abs(y_real))
        if denominador == 0:
            return np.nan
        return (np.sum(np.abs(y_real - predicciones)) / denominador) * 100

    def evaluar_modelo(y_real, predicciones, nombre):
        mae = mean_absolute_error(y_real, predicciones)
        rmse = np.sqrt(mean_squared_error(y_real, predicciones))
        r2 = r2_score(y_real, predicciones)
        wape = calcular_wape(y_real, predicciones)
        mask_no_cero = y_real > 0
        if mask_no_cero.sum() > 0:
            mape = mean_absolute_percentage_error(y_real[mask_no_cero], predicciones[mask_no_cero]) * 100
        else:
            mape = np.nan
        return {"Modelo": nombre, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE (%)": mape, "WAPE (%)": wape}

    # Ejemplo 1: benchmarking de modelos lado a lado (equivalente a la
    # sección "Model Benchmarking" del notebook — ahí ARIMA/XGBoost vs.
    # Naive/Moving Average, acá Regresión Lineal/XGBoost vs. Media Móvil).
    resultados = pd.DataFrame([
        evaluar_modelo(y_test.values, pred_baseline_3, "Media móvil 3 meses"),
        evaluar_modelo(y_test.values, pred_lr, "Regresión Lineal"),
        evaluar_modelo(y_test.values, pred_xgb, "XGBoost (log1p)")
    ])

    resultados_prediccion = test_df[["fecha", "producto_id", "demanda", "categoria_abc", "variabilidad", "usar_ml"]].copy()
    resultados_prediccion["prediccion_xgboost"] = pred_xgb
    resultados_prediccion["prediccion_baseline_3m"] = pred_baseline_3
    resultados_prediccion["prediccion_final"] = np.where(
        resultados_prediccion["usar_ml"],
        resultados_prediccion["prediccion_xgboost"],
        resultados_prediccion["prediccion_baseline_3m"]
    )
    resultados_prediccion["metodo_usado"] = np.where(resultados_prediccion["usar_ml"], "XGBoost", "Media móvil")
    resultados_prediccion["error_final"] = resultados_prediccion["prediccion_final"] - resultados_prediccion["demanda"]

    mask_no_cero = resultados_prediccion["demanda"] > 0
    if mask_no_cero.sum() > 0:
        mape_final = mean_absolute_percentage_error(
            resultados_prediccion.loc[mask_no_cero, "demanda"],
            resultados_prediccion.loc[mask_no_cero, "prediccion_final"]
        ) * 100
    else:
        mape_final = np.nan

    wape_final = calcular_wape(
        resultados_prediccion["demanda"].values,
        resultados_prediccion["prediccion_final"].values
    )

    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    ruta_importancia = os.path.join(carpeta_resultados, "xgb_feature_importance.png")
    plt.figure(figsize=(10, 6))
    importance.head(15).sort_values().plot.barh()
    plt.title("XGBoost - Importancia de Características")
    plt.tight_layout()
    plt.savefig(ruta_importancia, bbox_inches="tight")
    plt.close()

    # ============================================================
    # 30. PRONÓSTICO REAL DE LOS PRÓXIMOS 3 MESES (recursivo)
    # ============================================================
    # EXTENSIÓN PROPIA — no viene de ninguno de los 4 ejemplos de
    # referencia del proyecto (ver conversación de metodología). Ningún
    # ejemplo pronostica varios pasos hacia adelante de forma recursiva;
    # esta sección extiende el enfoque de un solo paso (Ejemplo 1/3) a
    # un horizonte de 3 meses.
    #
    # A diferencia de la sección de evaluación (que mide el modelo
    # contra meses que YA pasaron), esta sección pronostica los 3
    # meses SIGUIENTES al último mes disponible en el CSV. Como esos
    # meses todavía no ocurrieron, no hay demanda real con la cual
    # comparar — por eso aquí no se calcula ningún error.
    #
    # Es un pronóstico RECURSIVO (multi-step): la predicción del mes 1
    # se usa como si fuera dato real para calcular los lags del mes 2,
    # la del mes 2 para el mes 3, etc. Es el enfoque estándar para
    # extender un modelo de un paso a varios pasos sin tener que
    # entrenar un modelo distinto por horizonte (con ~12 meses de
    # historia, partir los datos en más subconjuntos los haría aún
    # más chicos). El costo es que el error se puede ir acumulando de
    # un mes a otro.

    NOMBRES_MES_ES = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
    }

    HORIZONTE_MESES = 3
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

    pronostico_futuro = []

    for producto in sorted(df_mensual["producto_id"].unique()):

        historial_producto = (
            df_mensual[df_mensual["producto_id"] == producto]
            .sort_values("fecha")
        )

        demandas = list(historial_producto["demanda"].values)

        if len(demandas) < 3:
            # No hay suficiente historial para calcular lag_1..lag_3
            continue

        ultima_fila = historial_producto.iloc[-1]
        segmento = producto_comportamiento[
            producto_comportamiento["producto_id"] == producto
        ].iloc[0]

        for h in range(HORIZONTE_MESES):
            mes_objetivo = proximo_mes + pd.DateOffset(months=h)

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

            X_futuro = pd.DataFrame([fila_features])[feature_cols]

            pred_xgb_futuro = float(np.maximum(np.expm1(model.predict(X_futuro))[0], 0))
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
            demandas.append(pred_final_futuro)

    pronostico_futuro_df = pd.DataFrame(pronostico_futuro)

    ruta_pronostico_futuro = os.path.join(carpeta_resultados, "pronostico_futuro.csv")
    pronostico_futuro_df.to_csv(ruta_pronostico_futuro, index=False)

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

    # ============================================================
    # 31. PUNTOS DE REORDEN Y EOQ
    # ============================================================
    # Ejemplo 1 (SKU Demand Forecasting, sección "Inventory Reorder
    # Recommendations") + Ejemplo 2 (Samir Saci, reglas de reposición
    # basadas en EOQ considerando la variabilidad de la demanda): mismas
    # fórmulas de Safety Stock / Reorder Point / EOQ / cantidad acotada
    # por presupuesto (ver más abajo, con las variables renombradas al
    # español pero la misma estructura matemática).
    #
    # El promedio y la desviación de demanda para el punto de reorden
    # se calculan sobre el pronóstico de los próximos 3 meses (arriba),
    # no sobre el backtest histórico: para decidir cuánto pedir hoy
    # interesa lo que se espera vender, no lo bien que el modelo
    # acertó en el pasado.

    SERVICE_LEVEL_Z = 1.65
    lead_time_default_meses = 0.5
    WORKING_CAPITAL_BUDGET = 500_000
    n_productos_activos = df_model["producto_id"].nunique()
    presupuesto_por_producto = WORKING_CAPITAL_BUDGET / n_productos_activos

    # "Hoy" para este cálculo es el último mes con datos del CSV (que es
    # de cuándo es el inventario_actual que se está usando), no la fecha
    # real del sistema — así la fecha estimada de pedido es coherente
    # con la fecha del inventario, sin importar cuándo se corra el
    # sistema ni qué tan viejo sea el archivo subido.
    fecha_referencia = df_mensual["fecha"].max()
    DIAS_POR_MES = 30.44

    reorder_table = []
    for producto in sorted(df_model["producto_id"].unique()):
        producto_pronostico = pronostico_futuro_df[pronostico_futuro_df["producto_id"] == producto]
        if len(producto_pronostico) == 0:
            continue
        pred_producto = producto_pronostico["prediccion_final"].values
        demanda_promedio = np.mean(pred_producto)
        desviacion_demanda = np.std(pred_producto)
        fila_params = df_mensual[df_mensual["producto_id"] == producto].iloc[-1]

        lead_time_meses = (
            fila_params["lead_time_semanas"] / 4.345
            if "lead_time_semanas" in df_mensual.columns and not pd.isna(fila_params.get("lead_time_semanas"))
            else lead_time_default_meses
        )
        costo_pedido_producto = fila_params.get("costo_pedido", np.nan)
        porcentaje_mantenimiento = fila_params.get("porcentaje_costo_mantenimiento", np.nan)
        costo_unitario_producto = fila_params.get("costo_unitario", np.nan)

        # Ejemplo 1: Safety Stock = Z * sigma_demanda * sqrt(L)
        stock_seguridad = SERVICE_LEVEL_Z * desviacion_demanda * np.sqrt(lead_time_meses)
        # Ejemplo 1: Reorder Point = d_promedio * L + Safety Stock
        punto_reorden = demanda_promedio * lead_time_meses + stock_seguridad
        inventario_actual = fila_params.get("inventario", np.nan)

        # Ejemplo 1: EOQ = sqrt(2 * D_anual * costo_pedido / costo_mantenimiento)
        demanda_anual_estimada = demanda_promedio * 12
        if not pd.isna(costo_unitario_producto) and not pd.isna(porcentaje_mantenimiento) and not pd.isna(costo_pedido_producto):
            costo_mantenimiento_unitario = costo_unitario_producto * porcentaje_mantenimiento
            eoq = np.sqrt((2 * demanda_anual_estimada * costo_pedido_producto) / costo_mantenimiento_unitario) if costo_mantenimiento_unitario > 0 else np.nan
        else:
            eoq = np.nan

        if not pd.isna(costo_unitario_producto) and costo_unitario_producto > 0:
            limite_presupuesto = int(presupuesto_por_producto / costo_unitario_producto)
        else:
            limite_presupuesto = np.nan

        # Ejemplo 1: cantidad a pedir acotada por el presupuesto de
        # capital de trabajo (min(EOQ, límite por presupuesto)).
        if not pd.isna(eoq) and not pd.isna(limite_presupuesto):
            cantidad_sugerida = min(eoq, limite_presupuesto)
        elif not pd.isna(eoq):
            cantidad_sugerida = eoq
        elif not pd.isna(inventario_actual):
            cantidad_sugerida = max(0, punto_reorden - inventario_actual)
        else:
            cantidad_sugerida = np.nan

        segmento = producto_comportamiento[producto_comportamiento["producto_id"] == producto].iloc[0]

        # Fecha estimada en la que el inventario va a llegar al punto de
        # reorden, asumiendo que se consume al ritmo promedio pronosticado
        # (una aproximación lineal — no sabe de picos puntuales dentro del
        # mes, pero da una fecha concreta en vez de solo un umbral de
        # cantidad). Si ya está en o por debajo del punto de reorden, la
        # fecha es "ya" (hoy mismo). Si no hay demanda pronosticada, no se
        # puede estimar cuándo se va a agotar.
        ya_hay_que_pedir = not pd.isna(inventario_actual) and inventario_actual <= punto_reorden
        if pd.isna(inventario_actual):
            fecha_estimada_pedido = None
            dias_para_pedido = np.nan
        elif ya_hay_que_pedir:
            fecha_estimada_pedido = fecha_referencia
            dias_para_pedido = 0
        elif demanda_promedio > 0:
            demanda_diaria = demanda_promedio / DIAS_POR_MES
            dias_para_pedido = (inventario_actual - punto_reorden) / demanda_diaria
            fecha_estimada_pedido = fecha_referencia + pd.Timedelta(days=dias_para_pedido)
        else:
            fecha_estimada_pedido = None
            dias_para_pedido = np.nan

        reorder_table.append({
            "producto_id": producto,
            "nombre_producto": segmento.get("nombre_producto"),
            "categoria": segmento.get("categoria"),
            "categoria_abc": segmento["categoria_abc"],
            "variabilidad": segmento["variabilidad"],
            "metodo_pronostico": "XGBoost" if segmento["usar_ml"] else "Media móvil",
            "demanda_promedio_mensual_pronosticada": round(demanda_promedio, 2),
            "lead_time_semanas": round(lead_time_meses * 4.345, 1),
            "lead_time_meses": round(lead_time_meses, 2),
            "stock_seguridad": round(stock_seguridad, 2),
            "punto_reorden": round(punto_reorden, 2),
            "inventario_actual": round(inventario_actual, 2) if not pd.isna(inventario_actual) else np.nan,
            "eoq": round(eoq, 0) if not pd.isna(eoq) else np.nan,
            "limite_presupuesto_unidades": limite_presupuesto,
            "cantidad_sugerida_pedido": round(cantidad_sugerida, 2) if not pd.isna(cantidad_sugerida) else np.nan,
            "fecha_estimada_pedido": fecha_estimada_pedido.strftime("%Y-%m-%d") if fecha_estimada_pedido is not None else None,
            "dias_para_pedido": round(dias_para_pedido) if not pd.isna(dias_para_pedido) else None,
            "ordenar": "SI" if ya_hay_que_pedir else "NO"
        })

    reorder_df = pd.DataFrame(reorder_table)

    ruta_modelo = os.path.join(carpeta_resultados, "modelo_xgboost_demanda.json")
    model.save_model(ruta_modelo)

    ruta_features = os.path.join(carpeta_resultados, "feature_columns.json")
    with open(ruta_features, "w") as f:
        json.dump(feature_cols, f)

    ruta_predicciones = os.path.join(carpeta_resultados, "predicciones.csv")
    resultados_prediccion.to_csv(ruta_predicciones, index=False)

    ruta_evaluacion = os.path.join(carpeta_resultados, "evaluacion_modelos.csv")
    resultados.to_csv(ruta_evaluacion, index=False)

    ruta_reorder = os.path.join(carpeta_resultados, "reorder_table.csv")
    reorder_df.to_csv(ruta_reorder, index=False)

    ruta_segmentacion = os.path.join(carpeta_resultados, "segmentacion_productos.csv")
    producto_comportamiento.to_csv(ruta_segmentacion, index=False)

    graficas_metricas = {}
    metricas = ["MAE", "RMSE", "MAPE (%)", "WAPE (%)"]
    for metrica in metricas:
        ruta_grafica = os.path.join(
            carpeta_resultados,
            f"comparacion_{metrica.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'porcentaje')}.png"
        )
        plt.figure(figsize=(8, 5))
        plt.bar(resultados["Modelo"], resultados[metrica])
        plt.title(f"Comparación de modelos - {metrica}")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(ruta_grafica, bbox_inches="tight")
        plt.close()
        graficas_metricas[metrica] = ruta_grafica

    return {
        "metricas": resultados.to_dict(orient="records"),
        "predicciones": resultados_prediccion.to_dict(orient="records"),
        "reorder": reorder_df.to_dict(orient="records"),
        "segmentacion": producto_comportamiento.to_dict(orient="records"),
        "mape_final": None if pd.isna(mape_final) else round(float(mape_final), 2),
        "wape_final": None if pd.isna(wape_final) else round(float(wape_final), 2),
        "pronostico_futuro": pronostico_futuro_df.to_dict(orient="records"),
        "pronostico_pivot": pronostico_pivot,
        "meses_pronosticados": meses_pronosticados,
        "meses_pronosticados_legibles": meses_pronosticados_legibles,
        "features_regresion_lineal": features_lr,
        "archivos": {
            "modelo": ruta_modelo,
            "features": ruta_features,
            "predicciones": ruta_predicciones,
            "evaluacion": ruta_evaluacion,
            "reorder": ruta_reorder,
            "segmentacion": ruta_segmentacion,
            "importancia": ruta_importancia,
            "segmentacion_grafica": ruta_segmentacion_grafica,
            "graficas": graficas_metricas,
            "pronostico_futuro": ruta_pronostico_futuro
        }
    }
