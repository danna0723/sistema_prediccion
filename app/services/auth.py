import json
import os
import uuid
from functools import wraps

from flask import current_app, flash, redirect, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash


def _cargar_usuarios():
    ruta = current_app.config["USUARIOS_PATH"]
    if not os.path.exists(ruta):
        return []
    with open(ruta, "r", encoding="utf-8") as f:
        usuarios = json.load(f)

    # Migración: cuentas creadas antes de que existiera el modelo
    # multiempresa no tienen empresa_id. Se les asigna uno acá (una
    # empresa propia, para no mezclarlas con la de otro admin) la
    # primera vez que se cargan después de esta actualización.
    faltantes = [u for u in usuarios if "empresa_id" not in u]
    if faltantes:
        for u in faltantes:
            u["empresa_id"] = uuid.uuid4().hex
        _guardar_usuarios(usuarios)

    return usuarios


def _guardar_usuarios(usuarios):
    ruta = current_app.config["USUARIOS_PATH"]
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(usuarios, f, ensure_ascii=False, indent=2)


def buscar_usuario(username):
    username = username.strip().lower()
    for u in _cargar_usuarios():
        if u["username"].lower() == username:
            return u
    return None


def buscar_por_email(email):
    email = email.strip().lower()
    for u in _cargar_usuarios():
        if u["email"].lower() == email:
            return u
    return None


def listar_usuarios(empresa_id):
    """Para la pantalla de administración de usuarios de una empresa —
    sin el hash de la contraseña, no hace falta ni conviene exponerlo.
    Solo devuelve las cuentas de esa empresa: cada admin ve y gestiona
    únicamente a sus propios empleados, no los de otros admins."""
    return [
        {"nombre": u["nombre"], "username": u["username"], "email": u["email"], "rol": u.get("rol", "empleado")}
        for u in _cargar_usuarios()
        if u.get("empresa_id") == empresa_id
    ]


def crear_usuario(nombre, username, email, password, rol="empleado", empresa_id=None):
    """
    Devuelve (usuario_creado, error). error es None si todo salió bien.
    No valida fuerza de contraseña ni formato de email más allá de que
    no estén vacíos — es un sistema de equipos internos, no un registro
    público con los mismos riesgos de un SaaS de cara al público.

    rol: "admin" o "empleado". El admin es quien sube los CSV y ve el
    panel técnico; un empleado solo ve demanda/inventario/presupuesto y
    puede confirmar pedidos (ver admin_required más abajo y las rutas
    protegidas en app/routes/main.py).

    empresa_id: a qué empresa (inventario) pertenece la cuenta. Si no
    se pasa uno (caso del registro público, que siempre crea un admin
    nuevo), se genera uno nuevo acá — es decir, cada admin que se
    registra arranca su propia empresa. Al crear un empleado desde
    /usuarios/nuevo, en cambio, se le pasa el empresa_id del admin que
    lo está creando, para que quede en la misma empresa que él.
    """
    if not nombre or not username or not email or not password:
        return None, "Todos los campos son obligatorios."
    if len(password) < 6:
        return None, "La contraseña debe tener al menos 6 caracteres."
    if rol not in ("admin", "empleado"):
        rol = "empleado"
    if buscar_usuario(username):
        return None, "Ese nombre de usuario ya está en uso."
    if buscar_por_email(email):
        return None, "Ese email ya está registrado."

    usuarios = _cargar_usuarios()
    usuario = {
        "nombre": nombre.strip(),
        "username": username.strip(),
        "email": email.strip(),
        "password_hash": generate_password_hash(password),
        "rol": rol,
        "empresa_id": empresa_id or uuid.uuid4().hex,
    }
    usuarios.append(usuario)
    _guardar_usuarios(usuarios)
    return usuario, None


def verificar_credenciales(username_o_email, password):
    """Permite iniciar sesión con nombre de usuario o email indistintamente."""
    usuario = buscar_usuario(username_o_email) or buscar_por_email(username_o_email)
    if usuario is None:
        return None
    if not check_password_hash(usuario["password_hash"], password):
        return None
    return usuario


def login_required(vista):
    """
    Protege una ruta: si no hay sesión iniciada, redirige a /login en vez
    de mostrar la página. No exige ningún rol en particular — la usan
    tanto el admin como los empleados.
    """
    @wraps(vista)
    def envoltura(*args, **kwargs):
        if "username" not in session:
            flash("Iniciá sesión para continuar.")
            return redirect(url_for("auth.login"))
        return vista(*args, **kwargs)
    return envoltura


def admin_required(vista):
    """
    Como login_required, pero además exige que el usuario logueado
    tenga rol "admin" — para las pantallas que un empleado no debería
    poder ver (subir/procesar CSV, panel técnico y sus descargas).
    Si un empleado intenta entrar, lo manda al dashboard en vez de
    mostrarle la página.
    """
    @wraps(vista)
    def envoltura(*args, **kwargs):
        if "username" not in session:
            flash("Iniciá sesión para continuar.")
            return redirect(url_for("auth.login"))
        if session.get("rol") != "admin":
            flash("Esa sección es solo para el administrador.")
            return redirect(url_for("main.dashboard"))
        return vista(*args, **kwargs)
    return envoltura
