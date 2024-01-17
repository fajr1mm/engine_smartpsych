from flask import Flask, request
import requests
import mysql.connector

app = Flask(__name__)

def save_to_database(data):
    try:
        # connect database
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password=None,
            database='engine_smartpsych'
        )

        cursor = connection.cursor()

        query = "INSERT INTO data_training (RESPONSE, LEVEL) VALUES (%s, %s)"

        for item in data['batch']:
            values = (item['RESPONSE'], item['LEVEL'])
            cursor.execute(query, values)

        connection.commit()

        cursor.close()
        connection.close()

        return jsonify({"message": "berhasil euyy"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update-data-training', methods=['POST'])
def update_data_training():
    try:
        data = request.get_json(force=True)

        # fungsi untuk save datanya
        response, status_code = save_to_database(data)

        return response, status_code

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan aplikasi Flask
    app.run(host='127.0.0.1', port=8080, debug=True)
