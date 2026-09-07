from flask import (
    Blueprint,
    render_template,
    request,
    current_app
)

from werkzeug.utils import secure_filename

import os

from app.services.sistema_prediccion import ejecutar_sistema


main_bp = Blueprint(
    "main",
    __name__
)


@main_bp.route("/")
def index():

    return render_template(
        "index.html"
    )


@main_bp.route(
    "/procesar",
    methods=["POST"]
)
def procesar():

    # Verificar que exista un archivo
    if "archivo" not in request.files:
        return "No se encontró ningún archivo"

    archivo = request.files["archivo"]

    # Verificar que tenga nombre
    if archivo.filename == "":
        return "No seleccionaste ningún archivo"

    # Validar CSV
    if not archivo.filename.lower().endswith(".csv"):
        return "Solo se permiten archivos CSV"

    # Nombre seguro
    nombre_archivo = secure_filename(
        archivo.filename
    )

    # Ruta donde se guardará
    ruta_archivo = os.path.join(
        current_app.config["UPLOAD_FOLDER"],
        nombre_archivo
    )

    # Guardar CSV
    archivo.save(
        ruta_archivo
    )

    try:

        # =====================================
        # EJECUTAR SISTEMA DE PREDICCIÓN
        # =====================================

        resultados_sistema = ejecutar_sistema(
            ruta_archivo
        )

        # =====================================
        # OBTENER RESULTADOS
        # =====================================

        resultados_modelos = resultados_sistema[
            "metricas"
        ]

        predicciones = resultados_sistema[
            "predicciones"
        ]

        reorden = resultados_sistema[
            "reorder"
        ]

        segmentacion = resultados_sistema[
            "segmentacion"
        ]

        mape_final = resultados_sistema[
            "mape_final"
        ]

        wape_final = resultados_sistema[
            "wape_final"
        ]

        # =====================================
        # MOSTRAR RESULTADOS
        # =====================================

        return render_template(

            "resultados.html",

            resultados_modelos=resultados_modelos,

            predicciones=predicciones[:20],

            reorden=reorden,

            segmentacion=segmentacion,

            mape_final=mape_final,

            wape_final=wape_final

        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )