from flask import Flask, request, jsonify

app = Flask(__name__)

EMITT_NAME = "Emitt"
EMITT_VERSION = "0.1.0"


@app.get("/")
def home():
    return jsonify({
        "name": EMITT_NAME,
        "version": EMITT_VERSION,
        "status": "online"
    })


@app.get("/api/status")
def status():
    return jsonify({
        "name": EMITT_NAME,
        "status": "online",
        "engine": "Emitt AI Engine",
        "engine_status": "not_installed"
    })


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({
            "error": "JSON body gerekli."
        }), 400

    message = data.get("message")

    if not isinstance(message, str):
        return jsonify({
            "error": "message alanı string olmalıdır."
        }), 400

    if not message.strip():
        return jsonify({
            "error": "Mesaj boş olamaz."
        }), 400

    # AI motorunu daha sonra buraya bağlayacağız.
    return jsonify({
        "model": EMITT_NAME,
        "message": message,
        "response": None,
        "status": "waiting_for_emitt_engine"
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=8000,
        debug=True
    )
