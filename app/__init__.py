import os
from flask import Flask


def create_app():

    app = Flask(__name__)
    app.config["SECRET_KEY"] = "cambia-esta-clave-por-una-propia"

    app.config["UPLOAD_FOLDER"] = os.path.join(os.getcwd(), "app", "uploads")
    app.config["RESULTADOS_FOLDER"] = os.path.join(os.getcwd(), "app", "resultados")

    # Aquí se guarda el resultado del último CSV procesado, para que
    # las pantallas de /desarrollo y /dashboard puedan leerlo sin
    # tener que volver a correr todo el pipeline cada vez que alguien
    # navega entre ellas.
    app.config["ULTIMO_RESULTADO_PATH"] = os.path.join(
        app.config["RESULTADOS_FOLDER"], "ultimo_resultado.json"
    )

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["RESULTADOS_FOLDER"], exist_ok=True)

    from app.routes.main import main_bp
    app.register_blueprint(main_bp)

    return app
