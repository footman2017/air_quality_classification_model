import paho.mqtt.client as mqtt
import json
import tensorflow as tf
import requests
import numpy as np

KATEGORI = {0: 'BAIK', 1: 'SEDANG', 2: 'TIDAK BAIK'}


def predict_with_model(json_data):
    # Ubah data JSON menjadi bentuk yang sesuai dengan model Anda
    # Sebagai contoh, model ini mengharapkan input berupa numpy array dengan bentuk (1, 6)
    input_data = [[json_data["pm10"], json_data["pm25"], json_data["so2"],
                   json_data["co"], json_data["o3"], json_data["no2"]]]

    # Load model Keras
    # Ganti dengan lokasi dan nama model Anda
    model = tf.keras.models.load_model('trained_model_lstm.h5')

    # Lakukan prediksi
    prediction = model.predict(input_data)
    highest_prob = np.max(prediction)
    highest_prob_index = np.argmax(prediction)
    predicted_class_label = KATEGORI[highest_prob_index]

    return {
        'accuracy': highest_prob,
        'category': predicted_class_label
    }

def on_message(client, userdata, message):      # Fungsi untuk melakukan subscribe ke topik MQTT
    payload = message.payload.decode('utf-8')
    data_json = json.loads(payload)

    # Lakukan prediksi dengan model
    result = predict_with_model(data_json)

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

client.loop_forever()

# data_json = {
#     "pm10": 30,
#     "pm25": 66,
#     "so2": 9,
#     "co": 13,
#     "o3": 1,
#     "no2": 7
# }

# print(predict_with_model(data_json))
