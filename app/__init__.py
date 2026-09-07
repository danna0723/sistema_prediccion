from flask import Flask
import os


def create_app():

    app = Flask(__name__)

    app.config["UPLOAD_FOLDER"] = os.path.join(
        os.getcwd(),
        "app",
        "uploads"
    )

    app.config["RESULTADOS_FOLDER"] = os.path.join(
        os.getcwd(),
        "app",
        "resultados"
    )

    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )

    os.makedirs(
        app.config["RESULTADOS_FOLDER"],
        exist_ok=True
    )

    from app.routes.main import main_bp

    app.register_blueprint(main_bp)

    return app