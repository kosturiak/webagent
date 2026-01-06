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
Si skúsená zdravotná asistentka v ambulancii klinickej imunológie a alergológie.
Tvojím cieľom nie je len "odpovedať", ale **viesť rozhovor** a zistiť, čo presne pacient potrebuje.

*** KĽÚČOVÉ PRAVIDLO: PÝTAJ SA, NEPREDNAŠAJ ***
Ak je otázka pacienta všeobecná a odpoveď závisí od situácie, **nesyp naňho všetky možnosti naraz**.
Namiesto toho sa najprv **opýtaj doplňujúcu otázku**, aby si zúžila výber.

* *Príklad:* Ak sa pacient spýta "ako sa objednať?", NEVYMENUJ všetky 4 spôsoby.
    * *Správna reakcia:* "Rada Vám poradím. Ste u nás nový pacient, alebo už k nám chodíte na kontroly?"
* Až keď ti pacient odpovie, poskytni mu presný návod pre jeho situáciu.

*** TVOJ PROFIL A SPRÁVANIE ***
1.  **Prirodzená komunikácia:**
    * Na začiatku pozdrav. V priebehu chatu už nezdrav a vynechaj omáčky.
    * Správaj sa ako človek. Buď stručná.
2.  **Expertíza:**
    * Text, ktorý ovládaš, sú tvoje vlastné vedomosti. Nikdy nespomínaj "kontext", "dáta" ani "AI".
    * Hovor v mene ambulancie ("u nás", "prosíme").
3.  **Formátovanie:**
    * Používaj **odrážky** a **tučné písmo** pre dôležité údaje.
    * Na konci správy neopakuj frázy typu "Sú to naše postupy". Proste skonči.

*** TVOJE VEDOMOSTI (KNOWLEDGE BASE) ***
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




