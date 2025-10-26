# Použi oficiálny základný obraz pre Python 3.11
FROM python:3.11-slim

# Nastav pracovný adresár vnútri kontajnera
WORKDIR /app

# Skopíruj "nákupný zoznam" a nainštaluj knižnice
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skopíruj zvyšok aplikácie (app.py a info.txt)
COPY . .

# Povedz Gunicornu, ako má spustiť tvoju app.py
# Bude počúvať na porte 8080, ktorý Cloud Run očakáva
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]