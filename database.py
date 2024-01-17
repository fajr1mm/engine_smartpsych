
from datetime import datetime
from flask import jsonify
from models import db, DataTraining, ModelVersi
from io import StringIO
import pandas as pd

def save_to_database(data):
    try:
        for item in data['batch']:
            current_time = datetime.now()
            versi=f"DT_v{current_time.strftime('%d%m%Y')}"
            response = item['RESPONSE']
            level = item.get('LEVEL', 0)  # Assuming LEVEL is in the data

            # Create a new data training entry with foreign key to the new version
            new_data_training = DataTraining(versi=versi, response=response, level=level, created_at=current_time)
            db.session.add(new_data_training)

        # Commit the changes to the database
        db.session.commit()

        return jsonify({"message": "Data training updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def save_csv_to_database(csv_content):
    try:
        # Baca CSV dari string
        df = pd.read_csv(StringIO(csv_content))

        for _, item in df.iterrows():
            current_time = datetime.now()
            versi = f"DT_v{current_time.strftime('%d%m%Y')}"
            response = item['RESPONSE']
            level = item.get('LEVEL', 0)  # Assuming LEVEL is in the data

            # Create a new data training entry with foreign key to the new version
            new_data_training = DataTraining(versi=versi, response=response, level=level, created_at=current_time)
            db.session.add(new_data_training)

        # Commit the changes to the database
        db.session.commit()

        return jsonify({"message": "Data training updated successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def fetch_selected_data(version):
    if version:
        # Fetch data based on the selected version and all previous versions
        selected_data = DataTraining.query.filter(DataTraining.versi <= version).all()
    else:
        # If version is not provided, fetch all data
        selected_data = DataTraining.query.all()

    return selected_data
    
def save_model_version(model_path):
    # Simpan versi model ke database
    new_model_version = ModelVersi(versi_model=f"ML_DT_{datetime.now().strftime('%d%m%Y')}", model_path=model_path)
    db.session.add(new_model_version)
    db.session.commit()

    return new_model_version.versi_model
