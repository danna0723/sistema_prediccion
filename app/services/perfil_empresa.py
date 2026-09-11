import json
import os

from flask import current_app


def _ruta_perfil(empresa_id):
    carpeta = os.path.join(current_app.config["EMPRESAS_FOLDER"], empresa_id)
    os.makedirs(carpeta, exist_ok=True)
    return os.path.join(carpeta, "perfil.json")


def cargar_nombre_empresa(empresa_id):
    """Devuelve el nombre que el admin le puso a su empresa, o None si
    todavía no lo ingresó — se usa para mostrarlo en el sidebar y en
    los reportes, y para precargarlo la próxima vez que suba un CSV."""
    ruta = _ruta_perfil(empresa_id)
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f).get("nombre_empresa")


def guardar_nombre_empresa(empresa_id, nombre_empresa):
    """Guarda (o actualiza) el nombre de la empresa. Se llama desde
    /procesar cada vez que se sube un archivo — así el campo queda
    editable sin necesitar una pantalla de configuración aparte."""
    nombre_empresa = (nombre_empresa or "").strip()
    if not nombre_empresa:
        return
    with open(_ruta_perfil(empresa_id), "w", encoding="utf-8") as f:
        json.dump({"nombre_empresa": nombre_empresa}, f, ensure_ascii=False)
