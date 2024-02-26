import whisper
import torch
import streamlit as st
from st_audiorec import st_audiorec
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import ConversationChain

from langchain.memory import ConversationBufferMemory
import os
from dotenv import load_dotenv
from pathlib import Path
import time
from pygame import mixer
from openai import OpenAI as op
import getans
load_dotenv(Path(".env"))

def audio_to_text():
    model = whisper.load_model("base")
    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        with open("/tmp/audio.wav", "wb") as f:
            f.write(wav_audio_data)
        data = whisper.load_audio("/tmp/audio.wav")
        
        result = model.transcribe(data)
        query = result["text"]
        return query
    
def querygen(query):
    reply=getans.answer(input=query)
    return reply


def text_to_speech(reply):
    speech=op()
    speech_file_path = "speech.mp3"
    response = speech.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=reply
    )
    response.stream_to_file(speech_file_path)
    return speech_file_path

def play_audio(file_path):
    mixer.init()
    mixer.music.load(file_path)
    mixer.music.play()
    while mixer.music.get_busy():  # wait for music to finish playing
        time.sleep(1)
    mixer.quit()




if __name__ == "__main__":
    st.title("GPT Based Psychotherapist")
    
    query=audio_to_text()
    st.write(query)
    if st.button("Generate Response"):
        reply=querygen(query)
        st.write(reply)
        file_path=text_to_speech(reply)
        play_audio(file_path)