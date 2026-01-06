import os
import vertexai
from vertexai.generative_models import GenerativeModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Nastavenie logovania, aby si videl chyby v Cloud Run logoch
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app) 

# --- Inicializácia Vertex AI ---
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = "europe-west1" 
vertexai.init(project=PROJECT_ID, location=LOCATION)

# --- Načítanie Vašej bázy znalostí ---
try:
    with open("info.txt", "r", encoding="utf-8") as f:
        KNOWLEDGE_BASE = f.read()
except FileNotFoundError:
    KNOWLEDGE_BASE = "Chyba: Informačný súbor nebol nájdený."
    logging.error("Súbor info.txt nebol nájdený!")

# --- Systémový Prompt ---
SYSTEM_PROMPT = f"""
Si "Asistent Ambulancie", nápomocný a empatický člen tímu v ambulancii.
Tvojou úlohou je odpovedať na otázky pacientov ohľadom organizácie, časeniek a objednávania.

*** PRÍSNE PRAVIDLÁ VYSTUPOVANIA (HERECKÝ MÓD) ***:
1. **Buď prirodzený:** Hovor v prvej osobe množného čísla ("u nás", "odporúčame", "naša ambulancia").
2. **ZÁKAZ technických rečí:**
   - NIKDY nepoužívaj slová ako: "v kontexte", "v poskytnutom texte", "podľa mojich inštrukcií", "mám v dátach", "som jazykový model".
   - ZLE: "V kontexte je uvedené, že volajte po 13:00."
   - DOBRE: "Prosím, volajte až po 13:00, vtedy vybavujeme telefonáty."
3. **Nevychádzaj z role:**
   - Ak sa pacient spýta "odkiaľ to vieš?", neodvolávaj sa na text. Povedz: "Sú to naše štandardné postupy v ambulancii."
   - Ak niečo nevieš, povedz: "Túto informáciu momentálne nemám, skúste zavolať." (Nehovor "nie je to v kontexte").
4. **Empatia:** Ak je pacient frustrovaný, buď chápavý.

*** TVOJE VEDOMOSTI (TOTO OVLÁDAŠ NASPAMÄŤ): ***
{KNOWLEDGE_BASE}
"""

# --- Inicializácia modelu ---
# Tu môžeš nechať 2.5 alebo zmeniť na "gemini-3.0-flash-preview" ak chceš novší model
model = GenerativeModel(
    "gemini-2.5-flash",
    system_instruction=SYSTEM_PROMPT
)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        
        if not data or "question" not in data:
            return jsonify({"error": "Chýbajúce dáta alebo kľúč 'question'"}), 400

        user_question = data.get("question")
        
        # 1. Získame históriu z JavaScriptu (ak neexistuje, použijeme prázdny zoznam)
        raw_history = data.get("history", [])
        
        # 2. Pripravíme zoznam správ pre Vertex AI
        vertex_messages = []

        # 3. Preformátujeme históriu z JavaScriptu do formátu pre Vertex AI
        # JavaScript posiela: { "role": "user", "content": "..." }
        # Vertex chce: { "role": "user", "parts": [{ "text": "..." }] }
        for msg in raw_history:
            role = msg.get("role")
            content = msg.get("content")
            if role and content:
                vertex_messages.append({
                    "role": role,
                    "parts": [{"text": content}]
                })

        # 4. Na koniec pridáme AKTUÁLNU otázku
        vertex_messages.append({
            "role": "user",
            "parts": [{"text": user_question}]
        })
        
        # 5. Zavoláme Gemini s celou históriou
        response = model.generate_content(
            vertex_messages,
            generation_config={"temperature": 0.0}
        )
        
        ai_answer = response.text

        return jsonify({"answer": ai_answer})

    except Exception as e:
        logging.error(f"Chyba v chate: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


