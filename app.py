import os
import vertexai
from vertexai.generative_models import GenerativeModel
from flask import Flask, request, jsonify
from flask_cors import CORS # Dôležité pre prepojenie s webom

# Inicializácia Flask aplikácie
app = Flask(__name__)
# Povolenie CORS, aby sa Váš web mohol pripojiť k tomuto API
CORS(app) 

# --- Inicializácia Vertex AI ---
# Tieto premenné si Cloud Run zoberie automaticky z prostredia
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "europe-west1" # Alebo Váš región
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Načítanie modelu (stačí raz pri štarte)
model = GenerativeModel("gemini-2.5-flash")

# Načítanie Vašej bázy znalostí
try:
    with open("info.txt", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = f.read()
except FileNotFoundError:
    KNOWLEDGE_BASE = "Chyba: Informačný súbor nebol nájdený."

# --- Systémový Prompt (Srdce agenta) ---
# Toto je inštrukcia, ktorá "uzamkne" AI iba na Vaše dáta
SYSTEM_PROMPT = f"""
Si "Asistent Ambulancie", priateľský a profesionálny chatbot imunologickej ambulancie.
Tvojou jedinou úlohou je odpovedať na otázky pacientov.
Odpovedaj VÝHRADNE na základe informácií z poskytnutého KONTEXTU.
NIKDY si nevymýšľaj informácie, ktoré nie sú v KONTEXTE.
Ak sa informácia v KONTEXTE nenachádza, slušne odpovedz, že túto informáciu nemáš k dispozícii a odkáž pacienta na telefonický kontakt.
Odpovedaj stručne a jasne.

--- KONTEXT ---
{KNOWLEDGE_BASE}
--- KONIEC KONTEXTU ---
"""

# --- API Endpoint (/chat) ---
# Toto bude adresa, ktorú bude volať Váš web
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Získanie otázky od pacienta (poslaná z webu ako JSON)
        data = request.get_json()
        user_question = data.get("question")

        if not user_question:
            return jsonify({"error": "Chýba otázka."}), 400

        # Zostavenie finálneho promptu
        # Kombinujeme systémovú inštrukciu (s kontextom) a otázku pacienta
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question}
        ]

        # Zavolanie Gemini API (presne ako to už poznáte)
        response = model.generate_content(
            messages,
            generation_config={"temperature": 0.0} # Nulová teplota pre faktické odpovede
        )
        
        ai_answer = response.text

        # Vrátenie odpovede Vášmu webu
        return jsonify({"answer": ai_answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Spustenie servera (toto je potrebné pre Cloud Run)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))