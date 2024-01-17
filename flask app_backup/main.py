# app1.py

from flask import Flask, request, jsonify, render_template
import requests
from fastapi_app import InputItem, PredictionItem

app = Flask(__name__)

# URL API FastAPI
# API_URL = 'https://3570-2404-8000-1024-3bd-d954-1eca-14f7-de03.ngrok-free.app/predict'
API_URL = 'http://127.0.0.1:8000/predict'


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
       # Ambil data dari permintaan POST
        data = request.get_json(force=True)
        input_batch = data['batch']

       # Ubah struktur data sesuai dengan skema InputItem
        input_items = [
            InputItem(id=item['id'], dimensi=item['dimensi'], jawaban=item['jawaban'])
            for item in input_batch
        ]

        # print(input_items)
        response = requests.post(API_URL, json=[item.dict() for item in input_items])
        predictions = response.json()
        print(predictions)
        print("cek")
        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Jalankan aplikasi Flask
    app.run(host='127.0.0.1', port=8080, debug=True)
