import streamlit as st
import time
import speech_recognition as sr
import pyttsx3
import requests
import json
from streamlit_webrtc import WebRtcMode, webrtc_streamer

webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

r = sr.Recognizer()
r.pause_threshold = 5


API_TOKEN = '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'
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


#Function to convert text to
# speech
def SpeakText(command):
     
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

MIN_Length = 30
MAX_Length = 200
MODEL = "base"
MyText = ""
summarize_text = ""
TIME_TO_RECO = 0
TIME_TO_SUMM = 0
audio = None

# Using "with" notation
with st.sidebar:
    MODEL = st.radio(
        "Choose a transcription model type:",
        ("tiny", "base", "small", "medium"),
        index = 1
    )

    st.divider() 

    st.write('How much do you want the percentage of the summary text from the original text?')
    MIN_Length, MAX_Length = st.slider(
    'Select a range of values',
    0, 100, (10, 30))
    st.write(f'[{MIN_Length}, {MAX_Length}]')



st.title("Summarize the doctor's conversations")
st.subheader('Record your voice or conversation, and we will summarize it for you')


with sr.Microphone(device_index=0) as source2:
    if st.button('Start'):
        st.subheader('Results')
        with st.spinner('Recording...'):
            # MyText, summarize_text, TIME_TO_RECO, TIME_TO_SUMM = run_func()

            #------------------
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)

            #listens for the user's input
            audio = r.listen(source2)
            #------------------
            st.success('Recording done!')

    else:
        st.write('Start Recording by click (Start) button')

st.divider() 

if audio is not None:
    with st.spinner('Prepare Text and summary...'):
        t1 = time.time()
        MyText = r.recognize_whisper(audio, model=MODEL)
        t2 = time.time()
        TIME_TO_RECO = t2 - t1 
        MyText = MyText.lower()

        try:
            st.subheader('Your Text:')
            st.markdown(str(MyText))

            st.markdown('Time to extract speak = ' + str(int(TIME_TO_RECO)) + 's' )

            st.divider() 
        except:
            st.subheader('Sorry, An error occurred when extract text from record !')


        t1 = time.time()
        summarize_text = query(
            {
                "inputs": MyText,
                "parameters": {"min_length": int((int( (MIN_Length/100)*len(MyText.split(' ')))*1.4)) , "max_length": int((int( (MAX_Length/100)*len(MyText.split(' ')))*1.4))  },
            }
        )
        t2 = time.time()
        TIME_TO_SUMM = t2 - t1 

        try:
            st.subheader('Summarized Text:')
            st.markdown(str(summarize_text[0]['summary_text']))

            st.markdown('Time to summarize = ' + str(int(TIME_TO_SUMM)) + 's')

            st.divider() 
        except:
            st.subheader('Sorry, An error occurred when summarize text !')


        st.success('All Done!!')

