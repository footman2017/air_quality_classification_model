import paho.mqtt.client as mqtt
import json


def publish_mqtt(json_data):
    client = mqtt.Client()
    broker_address = "test.mosquitto.org"
    # broker_address = "localhost"  # Ganti dengan alamat MQTT broker Anda
    # port = 8883  # Port for MQTT secure (TLS/SSL)
    port = 1883    # Port for MQTT secure (TLS/SSL)
    client.connect(broker_address, port)

    topic = "air_parameter"  # Ganti dengan nama topik yang Anda inginkan
    payload = json.dumps(json_data)
    client.publish(topic, payload)

    client.disconnect()


data_json = {
    "pm10": 66,
    "pm25": 90,
    "so2": 52,
    "co": 44,
    "o3": 37,
    "no2": 53
}

publish_mqtt(data_json)