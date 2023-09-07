import paho.mqtt.client as mqtt
import json
import tensorflow as tf
import requests
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib

KATEGORI = {0: 'BAIK', 1: 'SEDANG', 2: 'TIDAK SEHAT'}

# Load model Keras
# Ganti dengan lokasi dan nama model Anda
# model = tf.keras.models.load_model('trained_model_lstm.h5')


# def predict_with_model(json_data):
#     # Ubah data JSON menjadi bentuk yang sesuai dengan model Anda
#     # Sebagai contoh, model ini mengharapkan input berupa numpy array dengan bentuk (1, 6)
#     input_data = [[json_data["pm10"], json_data["pm25"], json_data["so2"],
#                    json_data["co"], json_data["o3"], json_data["no2"]]]

#     # Lakukan prediksi
#     prediction = model.predict(input_data)
#     highest_prob = np.max(prediction)
#     highest_prob_index = np.argmax(prediction)
#     predicted_class_label = KATEGORI[highest_prob_index]

#     return {
#         'accuracy': round((highest_prob * 100), 2),
#         # 'category': predicted_class_label
#         'category': highest_prob_index
#     }

# Load TabNet model
# It seems you've already loaded the model into loaded_clf
loaded_clf = TabNetClassifier()
loaded_clf.load_model("./tabnet_model_full_dataset.zip")
scaler = joblib.load('scaler.pkl')


def predict_with_model(json_data):
    # Convert JSON data to a format suitable for your model
    input_data = np.array([[json_data["pm10"], json_data["pm25"], json_data["so2"],
                            json_data["co"], json_data["o3"], json_data["no2"]]])

    # Preprocess the data: normalization, etc. Here, assuming you still have the 'scaler' object available from above.
    input_data = scaler.transform(input_data)

    # Make a prediction with the TabNet model
    probabilities = loaded_clf.predict_proba(input_data)
    highest_prob_index = np.argmax(probabilities)
    highest_prob = probabilities[0][highest_prob_index]

    predicted_class_label = KATEGORI[highest_prob_index]

    return {
        'accuracy': round((highest_prob * 100), 2),
        'category': highest_prob_index,
        'category_name': predicted_class_label,
    }


def send_to_be(json_data, prediction_result):
    url = "http://127.0.0.1:8000/api/store-prediction"

    payload = json.dumps({
        "pm10": json_data["pm10"],
        "pm25": json_data["pm25"],
        "so2": json_data["so2"],
        "co": json_data["co"],
        "o3": json_data["o3"],
        "no2": json_data["no2"],
        "location": json_data["location"],
        "prediction_result": int(prediction_result['category']),
        "accuracy": prediction_result['accuracy']
    })

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)

# Fungsi untuk melakukan subscribe ke topik MQTT


def on_message(client, userdata, message):
    payload = message.payload.decode('utf-8')
    data_json = json.loads(payload)

    # Lakukan prediksi dengan model
    result = predict_with_model(data_json)

    # simpan ke be
    response = send_to_be(data_json, result)

    # Lakukan sesuatu dengan hasil prediksi, misalnya mencetaknya
    print("Hasil prediksi:", result)


client = mqtt.Client()
# broker_address = "localhost"  # Ganti dengan alamat MQTT broker Anda
broker_address = "test.mosquitto.org"  # Ganti dengan alamat MQTT broker Anda
# port = 8883  # Ganti dengan port yang digunakan oleh MQTT broker Anda
port = 1883    # Ganti dengan port yang digunakan oleh MQTT broker Anda
client.connect(broker_address, port)

topic = "air_parameter"  # Ganti dengan nama topik yang Anda gunakan
client.subscribe(topic)
client.on_message = on_message
print('Subscriber ready')
client.loop_forever()