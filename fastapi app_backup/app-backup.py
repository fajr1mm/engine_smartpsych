from flask import Flask, request, jsonify, render_template
import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load your pickled model
with open('t5_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load your pickled tokenizer
with open('t5_tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

@app.route('/')
def index():
    return render_template('index.html')   

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari permintaan POST
        data = request.get_json(force=True)
        # print(data)
        input_batch = data['batch']

        # Lakukan prediksi untuk setiap input dalam batch
        API_URL = 'https://be87-125-163-7-230.ngrok-free.app'
        predictions = []
        for input_item in input_batch:
            input_text = f"{input_item['id']} {input_item['dimensi']} {input_item['jawaban']}"

            # Tokenisasi teks input
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

            # Lakukan inferensi menggunakan model
            output = model.generate(**inputs)
            prediction = tokenizer.decode(output[0], skip_special_tokens=True)

            # Tambahkan hasil prediksi ke daftar
            predictions.append({'id': input_item['id'], 'label': prediction})

        # print(predictions)
        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Jalankan aplikasi Flask
    app.run(host='127.0.0.1', port=8080, debug=True)
    # app1.py
