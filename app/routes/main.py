import os
import json

from flask import (
    Blueprint, render_template, request, redirect,
    url_for, flash, current_app, send_from_directory
)
from werkzeug.utils import secure_filename

from app.services.sistema_prediccion import ejecutar_sistema

main_bp = Blueprint("main", __name__)

EXTENSIONES_PERMITIDAS = {"csv"}


def extension_permitida(nombre_archivo):
    return (
        "." in nombre_archivo
        and nombre_archivo.rsplit(".", 1)[1].lower() in EXTENSIONES_PERMITIDAS
    )


def guardar_ultimo_resultado(resultado):
    """Guarda el resultado del procesamiento para que /desarrollo y
    /dashboard puedan leerlo después, sin volver a correr el pipeline.

    default=str convierte automáticamente tipos que json no conoce
    (Timestamps de pandas, tipos numpy, etc.) a texto, sin tener que
    tocar sistema_prediccion.py para forzar tipos nativos de Python.
    """
    ruta = current_app.config["ULTIMO_RESULTADO_PATH"]
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, default=str)


def cargar_ultimo_resultado():
    """Devuelve el último resultado guardado, o None si todavía no se
    ha procesado ningún CSV en esta sesión del servidor."""
    ruta = current_app.config["ULTIMO_RESULTADO_PATH"]
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


@main_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@main_bp.route("/procesar", methods=["POST"])
def procesar():
    archivo = request.files.get("archivo")

    if archivo is None or archivo.filename == "":
        flash("Debes seleccionar un archivo CSV antes de continuar.")
        return redirect(url_for("main.index"))

    if not extension_permitida(archivo.filename):
        flash("El archivo debe tener extensión .csv")
        return redirect(url_for("main.index"))

    nombre_seguro = secure_filename(archivo.filename)
    ruta_csv = os.path.join(current_app.config["UPLOAD_FOLDER"], nombre_seguro)
    archivo.save(ruta_csv)

    # Presupuesto opcional: si el usuario lo deja vacío o pone algo
    # inválido, ejecutar_sistema cae a su valor por defecto en vez de
    # bloquear la subida por esto.
    presupuesto_texto = request.form.get("presupuesto", "").strip()
    presupuesto = None
    if presupuesto_texto:
        try:
            presupuesto = float(presupuesto_texto)
            if presupuesto <= 0:
                presupuesto = None
        except ValueError:
            presupuesto = None

    try:
        resultado = ejecutar_sistema(
            ruta_csv, current_app.config["RESULTADOS_FOLDER"],
            presupuesto_capital_trabajo=presupuesto
        )
    except ValueError as e:
        flash(str(e))
        return redirect(url_for("main.index"))
    except Exception as e:
        flash(f"Ocurrió un error inesperado procesando el archivo: {e}")
        return redirect(url_for("main.index"))

    guardar_ultimo_resultado(resultado)

    # Después de procesar, se manda directo al dashboard (la pantalla
    # que vería un usuario real). La pantalla técnica queda disponible
    # aparte, en /desarrollo, para quien la necesite.
    return redirect(url_for("main.dashboard"))


@main_bp.route("/desarrollo", methods=["GET"])
def desarrollo():
    """Pantalla técnica: comparación de modelos, métricas, importancia
    de características, segmentación. Pensada para desarrollo/validación,
    no para el usuario final del sistema."""

    resultado = cargar_ultimo_resultado()

    if resultado is None:
        flash("Todavía no has subido ningún archivo. Sube un CSV primero.")
        return redirect(url_for("main.index"))

    return render_template("desarrollo.html", r=resultado)


@main_bp.route("/dashboard", methods=["GET"])
def dashboard():
    """Pantalla real del sistema: solo la demanda pronosticada del
    próximo mes por producto, sin tecnicismos de modelos ni métricas."""

    resultado = cargar_ultimo_resultado()

    if resultado is None:
        flash("Todavía no has subido ningún archivo. Sube un CSV primero.")
        return redirect(url_for("main.index"))

    return render_template("dashboard.html", r=resultado)


@main_bp.route("/descargar/<nombre_archivo>")
def descargar(nombre_archivo):
    return send_from_directory(
        current_app.config["RESULTADOS_FOLDER"],
        nombre_archivo,
        as_attachment=True
    )


@main_bp.route("/imagenes/<nombre_archivo>")
def imagenes(nombre_archivo):
    return send_from_directory(
        current_app.config["RESULTADOS_FOLDER"],
        nombre_archivo
    )
