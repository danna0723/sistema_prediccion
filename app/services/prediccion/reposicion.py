import numpy as np
import pandas as pd

SERVICE_LEVEL_Z = 1.65
LEAD_TIME_DEFAULT_MESES = 0.5
WORKING_CAPITAL_BUDGET_DEFAULT = 500_000
DIAS_POR_MES = 30.44


# ============================================================
# PUNTOS DE REORDEN Y EOQ
# ============================================================
# Ejemplo 1 (SKU Demand Forecasting, sección "Inventory Reorder
# Recommendations") + Ejemplo 2 (Samir Saci, reglas de reposición
# basadas en EOQ considerando la variabilidad de la demanda): mismas
# fórmulas de Safety Stock / Reorder Point / EOQ / cantidad acotada por
# presupuesto (con las variables renombradas al español pero la misma
# estructura matemática — ver comentarios línea a línea más abajo).
#
# El promedio y la desviación de demanda para el punto de reorden se
# calculan sobre el pronóstico de los próximos 3 meses, no sobre el
# backtest histórico: para decidir cuánto pedir hoy interesa lo que se
# espera vender, no lo bien que el modelo acertó en el pasado.
def calcular_tabla_reorden(
    df_model, df_mensual, producto_comportamiento, pronostico_futuro_df,
    presupuesto_capital_trabajo=None
):
    """
    presupuesto_capital_trabajo: monto total disponible para comprar
    inventario, ingresado por el usuario en el formulario de subida. Si
    no se especifica (None), se usa WORKING_CAPITAL_BUDGET_DEFAULT como
    respaldo — el mismo comportamiento que tenía el sistema antes de que
    el usuario pudiera ingresar su propio presupuesto.
    """
    working_capital_budget = (
        presupuesto_capital_trabajo
        if presupuesto_capital_trabajo is not None and presupuesto_capital_trabajo > 0
        else WORKING_CAPITAL_BUDGET_DEFAULT
    )
    n_productos_activos = df_model["producto_id"].nunique()
    presupuesto_por_producto = working_capital_budget / n_productos_activos

    # "Hoy" para este cálculo es el último mes con datos del CSV (que es
    # de cuándo es el inventario_actual que se está usando), no la fecha
    # real del sistema — así la fecha estimada de pedido es coherente
    # con la fecha del inventario, sin importar cuándo se corra el
    # sistema ni qué tan viejo sea el archivo subido.
    fecha_referencia = df_mensual["fecha"].max()

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
            else LEAD_TIME_DEFAULT_MESES
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

        if not pd.isna(eoq):
            cantidad_sugerida = eoq
        elif not pd.isna(inventario_actual):
            cantidad_sugerida = max(0, punto_reorden - inventario_actual)
        else:
            cantidad_sugerida = np.nan

        # Ejemplo 1: cantidad a pedir acotada por el presupuesto de
        # capital de trabajo (min(EOQ, límite por presupuesto)). Se aplica
        # sobre CUALQUIER método usado arriba (EOQ o la diferencia contra
        # el punto de reorden), no solo cuando hay EOQ: muchos catálogos
        # reales (como el de la clínica) no traen costo_pedido ni
        # porcentaje_costo_mantenimiento, así que el EOQ nunca se puede
        # calcular — pero sí se conoce el costo_unitario, y el presupuesto
        # tiene que respetarse igual.
        if not pd.isna(cantidad_sugerida) and not pd.isna(limite_presupuesto):
            cantidad_sugerida = min(cantidad_sugerida, limite_presupuesto)

        # Costo estimado de hacer ese pedido (cantidad * costo unitario).
        # No viene de ningún ejemplo de referencia — se agrega para que el
        # usuario vea en dinero, no solo en unidades, cuánto implica cada
        # pedido sugerido, y pueda contrastarlo contra su presupuesto.
        if not pd.isna(cantidad_sugerida) and not pd.isna(costo_unitario_producto):
            costo_estimado_pedido = cantidad_sugerida * costo_unitario_producto
        else:
            costo_estimado_pedido = np.nan

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
            "costo_unitario": round(costo_unitario_producto, 2) if not pd.isna(costo_unitario_producto) else np.nan,
            "costo_estimado_pedido": round(costo_estimado_pedido, 2) if not pd.isna(costo_estimado_pedido) else np.nan,
            "fecha_estimada_pedido": fecha_estimada_pedido.strftime("%Y-%m-%d") if fecha_estimada_pedido is not None else None,
            "dias_para_pedido": round(dias_para_pedido) if not pd.isna(dias_para_pedido) else None,
            "ordenar": "SI" if ya_hay_que_pedir else "NO"
        })

    return pd.DataFrame(reorder_table), working_capital_budget, presupuesto_por_producto
