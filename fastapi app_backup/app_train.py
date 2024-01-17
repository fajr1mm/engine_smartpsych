# app.py
from flask import Flask, request, jsonify
import mysql.connector
from sklearn.model_selection import train_test_split

app = Flask(__name__)
API_URL = "http://localhost:8000/train"

def fetch_database():
    # Konfigurasi koneksi ke database
    db_config = {
        'host': 'engin',
        'user': 'nama_pengguna',
        'password': 'kata_sandi',
        'database': 'nama_database',
    }

    conn = mysql.connector.connect(*db_config)

    cursor = conn.cursor()

    query = "SELECT FROM nama_tabel"

    cursor.execute(query)

    results = cursor.fetchall()

    cursor.close()
    conn.close()

def split_test_train(data_train):
    RESPONSE = [entry['RESPONSE'] for entry in data_train]
    LEVEL = [entry['LEVEL'] for entry in data_train]

    X_train, X_test, y_train, y_test = train_test_split(RESPONSE, LEVEL, test_size=0.2, random_state=42)

@app.route('/train')
def train():
    data_train = fetch_database()
    split_train = split_test_train(data_train)

    response = requests.post(API_URL, json=split_train)

    return jsonify(response.json())



if __name__ == '__main__':
    # Jalankan aplikasi Flask
    app.run(host='127.0.0.1', port=8080, debug=True)
