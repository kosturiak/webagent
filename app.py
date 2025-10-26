import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# --- Inicializácia Vertex AI ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "europe-west1"
vertexai.init(project=PROJECT_ID, location=LOCATION)

model = GenerativeModel("gemini-2.5-flash")

# Načítanie Vašej bázy znalostí
try:
    with open("info.txt", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = f.read()
except FileNotFoundError:
    KNOWLEDGE_BASE = "Chyba: Informačný súbor nebol nájdený."

# --- Systémový Prompt (Srdce agenta) ---
SYSTEM_PROMPT = f"""
Si "Asistent Ambulancie", priateľský a profesionálny chatbot imunologickej ambulancie.
Tvojou jedinou úlohou je odpovedať na otázky pacientov.
Odpovedaj VÝHRADNE na základe informácií z poskytnutého KONTEXTU.
NIKDY si nevymýšlaj informácie, ktoré nie sú v KONTEXTE.
Ak sa informácia v KONTEXTE nenachádza, slušne odpovedz, že túto informáciu nemáš k dispozícii a odkáž pacienta na telefonický kontakt.
Odpovedaj stručne a jasne.

--- KONTEXT ---
{KNOWLEDGE_BASE}
--- KONIEC KONTEXTU ---
"""

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        
        # Overenie, či JSON dáta prišli a či obsahujú kľúč 'question'
        if not data or "question" not in data:
            return jsonify({"error": "Chýbajúce dáta alebo kľúč 'question' v JSON body."}), 400

        user_question = data.get("question")

        # --- TOTO JE OPRAVENÁ ČASŤ ---
        # Správny formát pre Vertex AI 'generate_content'
        
        # 1. Systémový prompt ide ako samostatný argument
        system_instruction = SYSTEM_PROMPT
        
        # 2. História chatu používa 'parts' a 'text'
        chat_history = [
             {"role": "user", "parts": [Part.from_text(user_question)]}
             # V budúcnosti sem môžete pridať aj históriu konverzácie
             # {"role": "model", "parts": [Part.from_text(predchadzajuca_odpoved_ai)]},
             # {"role": "user", "parts": [Part.from_text(nova_otazka)]}
        ]
        
        # Zavolanie Gemini API v správnom formáte
        response = model.generate_content(
            chat_history,
            system_instruction=system_instruction, # <-- OPRAVA 1
            generation_config={"temperature": 0.0}
        )
        # --- KONIEC OPRAVENEJ ČASTI ---
        
        ai_answer = response.text

        return jsonify({"answer": ai_answer})

    except Exception as e:
        # Vrátime presnú chybu, ktorú vrátil Google, pre lepšie ladenie
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
