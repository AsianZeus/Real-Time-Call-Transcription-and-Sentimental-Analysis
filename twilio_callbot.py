import os
import base64
import json
import logging
from requests.adapters import Response
import speech_recognition as sr
from flask import Flask
from flask_sockets import Sockets
from pydub import AudioSegment
from twilio.rest import Client
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
from indictrans import Transliterator
import time

for i in os.listdir("Audio"):
    os.remove(f"Audio/{i}") if os.path.exists("recording.wav") else None

trn = Transliterator(source='eng', target='hin', build_lookup=True)
mode = "rohanrajpal/bert-base-codemixed-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(mode)
model = AutoModelForSequenceClassification.from_pretrained(mode)

app = Flask(__name__)
sockets = Sockets(app)

HTTP_SERVER_PORT = 8080
RAW_AUDIO_FILE_EXTENSION = "ulaw"
CONVERTED_AUDIO_FILE_EXTENSION = "wav"
ACC_SID = "XXXXXXXXX"
AUTH_TOKEN = "XXXXXXXXX"
FROM_NUMBER = "XXXXXXXXX"
TO_NUMBER = "XXXXXXXXX"

account_sid = ACC_SID
auth_token = AUTH_TOKEN
client = Client(account_sid, auth_token)

ngrok_url = "XXXXXXXXX.ngrok.io"


class Sequence:
    def __init__(self):
        self.CALL_FLOW = 0
        self.responses = [f"""<Response>
        <Play>XXXXXXXXX.wav</Play>
        <Start> <Stream url="wss://{ngrok_url}/" /> </Start>
        <Pause length = "10"/>
        </Response>""",
                          f"""<Response>
        <Play>XXXXXXXXX.wav</Play>
        <Pause length = "10"/>
        </Response>""",
                          f"""<Response>
        <Play>XXXXXXXXX.wav</Play>
        <Pause length = "30"/>
        </Response>""",
                          f"""<Response>
        <Play>XXXXXXXXX.wav</Play>
        <Pause length = "30"/>
        </Response>""",
                          f"""<Response>
        <Play>XXXXXXXXX.wav</Play>
        <Pause length = "30"/>
        </Response>""",
                          f"""<Response>
        <Play>XXXXXXXXX.wav</Play>
        </Response>"""]

    def get_call_flow(self):
        return self.CALL_FLOW

    def get_response(self):
        try:
            tmp = self.responses[self.CALL_FLOW]
            self.CALL_FLOW += 1
        except:
            print("exception in get response")
            tmp = self.responses[-1]
        return tmp


seq = Sequence()

call = client.calls.create(
    twiml=seq.get_response(), from_=FROM_NUMBER, to=TO_NUMBER)

call_sid = call.sid


r = sr.Recognizer()


def getSentiment(text):
    labels = ["Neutral", "Negative", "Positive"]
    try:
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
    except:
        text = text[:512]
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    sc = {}
    for i in range(3):
        sc[labels[i]] = np.round(float(scores[i]), 4)
    return 0 if sc["Negative"] > sc["Positive"] else 1


def recognize_speech(recording_audio_path):
    with sr.AudioFile(recording_audio_path) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio)
    except Exception as e:
        return None


def make_update(speech):
    hin = trn.transform(speech)
    sentiment = getSentiment(hin)
    CALL_FLOW = seq.get_call_flow()
    print(f"You said: {hin} | Sentiment: {sentiment} | Callflow: {CALL_FLOW}")
    if(CALL_FLOW == 2 or CALL_FLOW == 3 or CALL_FLOW == 4):
        call = client.calls(call_sid).update(
            twiml=seq.get_response())
    else:
        if sentiment:
            call = client.calls(call_sid).update(
                twiml=seq.get_response())
        else:
            call = client.calls(call_sid).update(
                twiml=seq.responses[-1])


class StreamAudioRecording:
    def __init__(self, audio_recording_path):
        self.audio_recording_path = audio_recording_path
        self.f = None
        self.audio_file_path = None
        self.data_buffer = b''
        self.data = []

    def start_recording(self, call_id):
        self.audio_file_path = os.path.join(
            self.audio_recording_path, f"{call_id}.{RAW_AUDIO_FILE_EXTENSION}")
        self.f = open(self.audio_file_path, 'wb')

    def write_buffer(self, buffer):
        self.data_buffer += buffer
        self.f.write(buffer)

    def append_buffer(self):
        self.data.append(self.data_buffer)
        self.data_buffer = b''

    def stop_recording(self):
        self.f.close()
        converted_audio_path = self.audio_file_path.replace(RAW_AUDIO_FILE_EXTENSION,
                                                            CONVERTED_AUDIO_FILE_EXTENSION)
        self.convert_call_recording(self.audio_file_path, converted_audio_path)
        return converted_audio_path

    @ staticmethod
    def convert_call_recording(mulaw_path, wav_path):
        new = AudioSegment.from_file(
            mulaw_path, "mulaw", frame_rate=8000, channels=1, sample_width=1)
        new.frame_rate = 8000
        new.export(wav_path, format="wav", bitrate="8k")


@ sockets.route('/')
def echo(ws):

    app.logger.info("Connection accepted")

    # A lot of messages will be sent rapidly. We'll stop showing after the first one.
    has_seen_media = False
    message_count = 1

    recording = StreamAudioRecording("/Users/ace/Desktop/Twilio/Audio")
    recording.start_recording("0")

    while not ws.closed:
        message = ws.receive()
        # print(f"Received message: {message}")
        if message is None:
            app.logger.info("No message received...")
            continue

        # Messages are a JSON encoded string
        data = json.loads(message)

        # Using the event type you can determine what type of message you are receiving
        if data['event'] == "connected":
            app.logger.info("Connected Message received: {}".format(message))
        if data['event'] == "start":
            app.logger.info("Start Message received: {}".format(message))
        if data['event'] == "media":
            payload = data['media']['payload']
            chunk = base64.b64decode(payload)
            recording.write_buffer(chunk)
            if(message_count % 58 == 0):
                recording.append_buffer()
                try:
                    rb1 = recording.data[-1].count(b'\xff')
                    if(rb1 > 7000):
                        st = time.time()
                        recording_audio_path = recording.stop_recording()
                        recording.start_recording(str(message_count))
                        speech = recognize_speech(recording_audio_path)
                        if speech:
                            print(time.time()-st)
                            make_update(speech)
                except:
                    pass
            message_count += 1
            if not has_seen_media:
                payload = data['media']['payload']
                chunk = base64.b64decode(payload)
                has_seen_media = True
        if data['event'] == "stop":
            app.logger.info("Stop Message received: {}".format(message))
            recording.append_buffer()
            break

    app.logger.info(
        "Connection closed. Received a total of {} messages".format(message_count))
    recording_audio_path = recording.stop_recording()

    recording_audio_path = recording.stop_recording()
    speech = recognize_speech(recording_audio_path)
    if speech:
        print("\n*****\n", speech, "\n******")


if __name__ == '__main__':
    app.logger.setLevel(logging.DEBUG)
    from gevent import pywsgi
    from geventwebsocket.handler import WebSocketHandler
    server = pywsgi.WSGIServer(
        ('', HTTP_SERVER_PORT), app, handler_class=WebSocketHandler)
    print("Server listening on: http://localhost:" + str(HTTP_SERVER_PORT))
    server.serve_forever()
