import streamlit as st
import time
# import speech_recognition as sr
import pyttsx3
import whisper
import librosa
import openai

st.set_page_config(page_title='Summary Doctor')

MIN_Length = 30
MAX_Length = 200
TRANSCRIPT_MODEL = "base"
GPT_MODEL = ''
MyText = ""
summarize_text = ""
TIME_TO_RECO = 0
TIME_TO_SUMM = 0
audio = None
API_KEY = ''

PROMPT = "write a doctor's letter with a clear action plan at the end, based on the following conversation:"

# Using "with" notation
with st.sidebar:
    TRANSCRIPT_MODEL = st.radio(
        "Choose a transcription model type:",
        ("tiny", "base", "small", "medium"),
        index = 1
    )

    st.divider() 

    API_KEY = st.text_input('Enter your OpenAI api key:')

    st.divider() 

    GPT_MODEL = st.radio(
        "Choose a GPT model type:",
        ("gpt-3.5-turbo", "gpt-4"),
        index = 0
    )

    st.divider() 

    PROMPT = st.text_area('Enter your prompt (or leave it empty to use default):')


st.title("Summarize the doctor's conversations")
st.subheader('Record your voice or conversation, and we will summarize it for you')



uploaded_file = st.file_uploader("Choose an audio file (.mp3, .wav)", accept_multiple_files=False,
                                type=['mp3', 'wav'])


def call_api(text, model_id):
    model_id = model_id
    conversation = []
    conversation.append({'role': 'user', 'content': text})
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation
    )
    return response.choices[0].message.content

if st.button("Extract & Generate") and uploaded_file is not None and API_KEY != '':
    openai.api_key = API_KEY
    st.divider()
    with st.spinner('Prepare Text and summary...'):

        audio, _ = librosa.load(uploaded_file, sr=16000)
        model = whisper.load_model(TRANSCRIPT_MODEL)

        t1 = time.time()
        MyText = model.transcribe(audio)
        t2 = time.time()
        TIME_TO_RECO = t2 - t1 
        MyText = MyText["text"]

        try:
            st.subheader('Your Text:')
            st.markdown(str(MyText))

            st.markdown('Time to extract speak = ' + str(int(TIME_TO_RECO)) + 's' )

            st.divider() 
        except:
            st.subheader('Sorry, An error occurred when extract text from record !')


        try:
            if PROMPT == '':
                PROMPT = "write a doctor's letter with a clear action plan at the end, based on the following conversation:"

            t1 = time.time()
            Final_Text = PROMPT + '\n' + MyText
            summarize_text = call_api(Final_Text, GPT_MODEL)
            t2 = time.time()
            TIME_TO_SUMM = t2 - t1
            st.subheader('Summarized Text:')
            st.markdown(str(summarize_text))

            st.markdown('Time to summarize = ' + str(int(TIME_TO_SUMM)) + 's')

            st.divider() 

            st.success('All Done!!')

        except:
            st.error('An error occurred when try make request, your api key may not be valid', icon="🚨") 

elif uploaded_file is None:
    st.warning('Please upload your file', icon="⚠️")
elif API_KEY == '':
    st.warning('Please enter your OpenAI api key', icon="⚠️")

