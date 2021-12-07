import json
import os
from flask import Flask
from flask import request
from flask import make_response
import random
import serial
import cv2
import time
import requests
import numpy as np
import tflite_runtime.interpreter as tflite
from threading import Thread
from detection import detection
import paho.mqtt.publish as publish # pip3 install paho-mqtt
import paho.mqtt.client as mqtt
import ssl
from gpiozero import Servo
from time import sleep
import RPi.GPIO as GPIO
from sendMSG import sendLINEMSG

servo = Servo(8)

GPIO.setmode(GPIO.BCM)             # choose BCM or BOARD  
GPIO.setup(7, GPIO.OUT)

port = 1883 # default port
Server_ip = "broker.netpie.io" 

Subscribe_Topic = "@msg/LED"
Publish_Topic = "@shadow/data/update"

Client_ID = "1a9283ae-4d75-4a12-84fd-2581965fbac1"
Token = "pW2SsuDa1VhrmdUEvjdt3uuMAWMApzdi"
Secret = "OCcpReZ1oc!K7zr_#WFOqw0w*xtpP!w3"

MqttUser_Pass = {"username":Token,"password":Secret}

DOOR_Status = "door_off"
PUMP_Status = "pump_off"
sensor_data = {"Ultra": 0, "Humi": 0, "DOOR" : DOOR_Status,"PUMP" : PUMP_Status}

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    client.subscribe(Subscribe_Topic)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global DOOR_Status
    global PUMP_Status
    print(msg.topic+" "+str(msg.payload))
    data_receive = msg.payload.decode("UTF-8")
    DOOR_Status = data_receive
    PUMP_Status = data_receive

client = mqtt.Client(protocol=mqtt.MQTTv311,client_id=Client_ID, clean_session=True)
client.on_connect = on_connect
client.on_message = on_message

client.subscribe(Subscribe_Topic)
client.username_pw_set(Token,Secret)
client.connect(Server_ip, port)
client.loop_start()


# Flask
app = Flask(__name__)
@app.route('/', methods=['POST']) 

def MainFunction():

    question_from_dailogflow_raw = request.get_json(silent=True, force=True)
    
    sensor_data["Ultra"] = float(distance())

        
    data_out=json.dumps({"data": sensor_data}) # encode object to JSON
    print(data_out)
    client.publish(Publish_Topic, data_out, retain= True)
    print ("Publish.....")
    time.sleep(2)

    answer_from_bot = generating_answer(question_from_dailogflow_raw)
    print('sukit')
    r = make_response(answer_from_bot)
    r.headers['Content-Type'] = 'application/json'

    return r



def generating_answer(question_from_dailogflow_dict):

    print(json.dumps(question_from_dailogflow_dict, indent=4 ,ensure_ascii=False))

    intent_group_question_str = question_from_dailogflow_dict["queryResult"]["intent"]["displayName"] 

    if intent_group_question_str == 'หิวจัง':
        answer_str = menu_recormentation()
    elif intent_group_question_str == 'ระยะ': 
        answer_str = distance()
    elif intent_group_question_str == 'เช็คของ':
        if float(distance()) > 35:
            answer_str = 'NO ITEM'
        else:
            answer_str = detect()
    elif intent_group_question_str == 'ออก':
        answer_str = 'Have a Great Day!'
        sendLINEMSG('Have a Great Day!')
        
        GPIO.output(7, 1)          
        sensor_data["PUMP"] = 'pump_on'
        sleep(0.2)
        
        data_out=json.dumps({"data": sensor_data})
        client.publish(Publish_Topic, data_out, retain= True)
        
        GPIO.output(7, 0) 
        sensor_data["PUMP"] = 'pump_off'
        sleep(1)
        
        data_out=json.dumps({"data": sensor_data})
        client.publish(Publish_Topic, data_out, retain= True)
        time.sleep(1)
        
        servo.max()
        sensor_data["DOOR"] = 'door_on'
        sleep(1)
        data_out=json.dumps({"data": sensor_data})
        client.publish(Publish_Topic, data_out, retain= True)

        
        servo.min()
        sensor_data["DOOR"] = 'door_off'
        sleep(1)
        data_out=json.dumps({"data": sensor_data})
        client.publish(Publish_Topic, data_out, retain= True)
        time.sleep(1)
        
    else: answer_str = "ผมไม่เข้าใจ คุณต้องการอะไร"

    answer_from_bot = {"fulfillmentText": answer_str}
    
    answer_from_bot = json.dumps(answer_from_bot, indent=4) 

    return answer_from_bot

def menu_recormentation():
    menu_name = 'ข้าวขาหมู'
    answer_function = menu_name + ' สิ น่ากินนะ'
    return answer_function


def distance():
    uart = serial.Serial('/dev/ttyACM0',115200,timeout=1)
    uart.close()
    uart.open()
    string_echo = ""
    while len(string_echo) < 1:
        string_echo = uart.read_until('\r'.encode()).decode('utf-8')
    print (string_echo)
    uart.close()
    answer_function = string_echo
    return answer_function

def detect():
    answer_function = detection()
    print(answer_function)
    return answer_function

    
    
#Flask
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port, host='0.0.0.0', threaded=True)

