from flask import Flask, request, jsonify, render_template
import requests
from models import db, DataTraining
from database import save_to_database, fetch_selected_data, save_model_version
from train_model import train_model, TrainItem
from modelpredict import InputItem, PredictionItem, predictmodel
from preprocessing_function import clean_text
from pydantic import BaseModel
from typing import List, Optional
from models import db, DataTraining
import pandas as pd


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/engine_smartpsych'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

API_URL = 'http://127.0.0.1:8001'
# Create tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

## PREDICT
@app.route('/predict', methods=['POST'])
def predict():
    try:
       # Ambil data dari permintaan POST
        data = request.get_json(force=True)
        input_batch = data.get('batch', [])
        save_dir = data.get('save_dir', None)

        # print(input_batch)

       # Ubah struktur data sesuai dengan skema InputItem
        input_items = [
            InputItem(id=item['id'], dimensi=item['dimensi'], jawaban=item['jawaban'])
            for item in input_batch
        ]

        input_predict = input_items

        response = predictmodel(save_dir, input_predict)
        predictions = response
        print(predictions)

        return predictions

    except Exception as e:
        return jsonify({'error': str(e)})
#------------------------------------------------------------------------------------------------------------#
## TRAIN 
@app.route('/train', methods=['POST'])
def train():
    try:
        data = request.get_json(force=True)
        columns = data.get('columns', [])
        parameters = data.get('parameters', {})
        version = data.get('version', None)

        if version:
            # Fetch data based on the selected version
            selected_data = fetch_selected_data(version)
            train_item_data = [{col: getattr(row, col) for col in columns} for row in selected_data]
            train_item = TrainItem(columns=columns, parameters={}, data=train_item_data)
            result = train_model(train_item, columns=columns, parameters=parameters)
        else:
            result = train_model(data, columns=columns, parameters=parameters)


        save_model_version(result['model_path'])

        return result

    except Exception as e:
        return jsonify({"error": str(e)}), 500
#------------------------------------------------------------------------------------------------------------#
## UPDATE DATA TRAINING DATABASE and CSV asdasdas
@app.route('/update-data-training', methods=['POST'])
def update_data_training():
    try:
        data = request.get_json(force=True)
        # Clean and process data
        for item in data['batch']:
            item['JAWABAN'] = clean_text(item['JAWABAN'])

        for item in data['batch']:
            item['RESPONSE'] = f"{item['DIMENSI']}; {item['JAWABAN']}"

        processed_data = {
            "batch": [
                {k: v for k, v in item.items() if k not in ["DIMENSI", "JAWABAN"]} for item in data["batch"]
            ]
        }
        # Save data to the database using the save_to_database function
        return save_to_database(processed_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/update-data-training-csv', methods=['POST'])
def update_data_training_csv():
    try:
        # Pastikan request memiliki file CSV
        if 'file' not in request.files:
            return jsonify({"error": "No CSV file provided"}), 400

        file = request.files['file']
        # Baca file CSV ke DataFrame
        df = pd.read_csv(file)

        processed_data = {
            "batch": [
                {k: v for k, v in item.items() if k not in ["DIMENSI", "JAWABAN"]} for _, item in df.iterrows()
            ]
        }

        # Simpan data ke database
        return save_to_database(processed_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
#------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    # Jalankan aplikasi Flask
    app.run(host='127.0.0.1', debug=True)
