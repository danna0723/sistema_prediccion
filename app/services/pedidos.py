import json
import os
from datetime import datetime

from flask import current_app


def _ruta_pedidos(empresa_id):
    carpeta = os.path.join(current_app.config["EMPRESAS_FOLDER"], empresa_id)
    os.makedirs(carpeta, exist_ok=True)
    return os.path.join(carpeta, "pedidos.json")


def cargar_pedidos(empresa_id):
    """Devuelve el registro de pedidos confirmados de una empresa:
    {producto_id: {...}}. Vive aparte de ultimo_resultado.json a
    propósito — el historial de "qué ya se pidió" no debería borrarse
    cada vez que se sube un CSV nuevo para actualizar el pronóstico."""
    ruta = _ruta_pedidos(empresa_id)
    if not os.path.exists(ruta):
        return {}
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def _guardar_pedidos(empresa_id, pedidos):
    ruta = _ruta_pedidos(empresa_id)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(pedidos, f, ensure_ascii=False, indent=2)


def registrar_pedido(empresa_id, producto_id, proveedor, cantidad, costo, usuario):
    """Marca un producto como pedido: guarda proveedor elegido, cantidad
    y costo (tomados del análisis vigente, no de lo que mande el
    formulario) junto con quién lo confirmó y cuándo."""
    pedidos = cargar_pedidos(empresa_id)
    pedidos[str(producto_id)] = {
        "proveedor": proveedor,
        "cantidad": cantidad,
        "costo": costo,
        "usuario": usuario,
        "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    _guardar_pedidos(empresa_id, pedidos)


def quitar_pedido(empresa_id, producto_id):
    """Deshace la confirmación de un pedido (por si se marcó por error)."""
    pedidos = cargar_pedidos(empresa_id)
    pedidos.pop(str(producto_id), None)
    _guardar_pedidos(empresa_id, pedidos)
