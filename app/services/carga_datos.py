# import numpy as np
# import pandas as pd
    
#     # ============================================================
#     # 1. CARGAR CSV
#     # ============================================================

# df = pd.read_csv(ruta_archivo)

# print("Dimensiones:", df.shape)
# print("\nColumnas originales:")
# print(df.columns.tolist())

# print("\nValores nulos:")
# print(df.isnull().sum())


# # ============================================================
# # 2. NORMALIZAR NOMBRES DE COLUMNAS
# # ============================================================

# df = df.rename(columns={

#     # FECHA
#     "Sale_Date": "fecha",
#     "Date": "fecha",
#     "date": "fecha",
#     "Fecha": "fecha",

#     # PRODUCTO
#     "Product_ID": "producto_id",
#     "Product": "producto_id",
#     "Producto": "producto_id",

#     # DEMANDA
#     "Quantity_Sold": "demanda",
#     "Sales": "demanda",
#     "Demand": "demanda",
#     "Demanda": "demanda",

#     # PRECIO
#     "Unit_Price": "precio",
#     "Price": "precio",
#     "Price_per_unit": "precio",

#     # INVENTARIO
#     "Inventario_Actual": "inventario",
#     "Inventory": "inventario",
#     "Stock": "inventario",

#     # TIENDA
#     "Store": "tienda_id",
#     "Store_ID": "tienda_id",

#     # CATEGORÍA
#     "Category": "categoria",
#     "Product_Category": "categoria",

#     # INVENTARIO / COSTOS
#     "Lead_time_Semana": "lead_time_semanas",
#     "Costo_pedido": "costo_pedido",
#     "Costo_unitario": "costo_unitario",
#     "Porcentaje_costo_mantenimiento": "porcentaje_costo_mantenimiento"
# })


# # ============================================================
# # 3. VALIDAR COLUMNAS OBLIGATORIAS
# # ============================================================

# required_columns = ["fecha","producto_id","demanda"]

# for col in required_columns:

#     if col not in df.columns:
#         raise ValueError(
#             f"Falta la columna obligatoria: {col}"
#         )


# # ============================================================
# # 4. LIMPIEZA DE DATOS
# # ============================================================

# df["fecha"] = pd.to_datetime(df["fecha"],errors="coerce")

# df["demanda"] = pd.to_numeric(
#     df["demanda"],
#     errors="coerce"
# )

# df = df.dropna(
#     subset=[
#         "fecha",
#         "producto_id",
#         "demanda"
#     ]
# )

# # La demanda no puede ser negativa

# df["demanda"] = np.maximum(
#     df["demanda"],
#     0
# )

# df = df.sort_values(
#     [
#         "producto_id",
#         "fecha"
#     ]
# ).reset_index(drop=True)


# print("\nDataset preparado:")
# print("Número de productos:", df["producto_id"].nunique())
# print("Fecha inicial:", df["fecha"].min())
# print("Fecha final:", df["fecha"].max())
# print("Demanda promedio:", df["demanda"].mean())
# print("Demanda máxima:", df["demanda"].max())

