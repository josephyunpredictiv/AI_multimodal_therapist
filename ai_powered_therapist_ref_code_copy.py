# -*- coding: utf-8 -*-
"""
Original file is located at. Not avalible publicly
    https://colab.research.google.com/drive/1AYC7ZmyBoXKbH8r6Ipam9Nk92wF8vBa5

## Outline
I will be referencing each layer as steps.

For example, 'User Input' is step 0 while 'Take picture' and 'Start recording' is step 1 etc..

## Step 0:
We assume this will be done in frontend and therefore, will not include this in the code

## Step 1:
Take picture
And start recording
"""

# Assuming that picture is taken then stored into image.png


# def autoRecording():
# Utilizing voice detection

!pip install webrtcvad
import webrtcvad
vad = webrtcvad.Vad()
vad.set_mode(3) # noise filterning, int 0 to 3 with 3 most agressive
sample_rate = 16000
frame_duration = 10  # ms, option of 10, 20, 30
frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
print('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))

#Some sort of while loop that will stop recording once it stops detecting sound for some set amount of time.


# We also assume that this outputs a file called audio.mp3

"""## Step 2:
1. Emotion Detection
2. Tone detection
3. Transcription
4. Sentiment analysis

Assuming that we have a recording file called **audio.mp3**  
And an image called **image.png**  
(from steps 0 and 1)
"""

# 1. Emotion detection

# We will be running the 'image_emotion_gender_demo.py' file with image.png
# Check github @ https://github.com/AregGevorgyan/WAICY2023/tree/main/emotion_detection_cleaned


# I will set a dummy variable called emotion so code is continuable
emotion = "Happy"

# 2. Tone detection
# Will need to put .mp3 to .wav

!pip install transformers

from transformers import Wav2Vec2ForSequenceClassification
import torchaudio
import torch

# Load the model
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Load and preprocess an audio file
audio_input, sample_rate = torchaudio.load("recordingsample.wav")

# Resample if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    audio_input = resampler(audio_input)

# Make sure the audio input is in the right shape: (batch_size, num_samples)
audio_input = audio_input.squeeze(0)  # Remove channel dimension if it exists
inputs = audio_input.unsqueeze(0)  # Add a batch dimension

# Predict emotion
with torch.no_grad():
    logits = model(inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# Output the predicted ID
print("Predicted Emotion ID:", predicted_ids.item())
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
print("Predicted Emotion: ", emotions[predicted_ids.item()])
tone_emotion = emotions[predicted_ids.item()]

# 3. Transcription

from openai import OpenAI
client = OpenAI(api_key="xxxxxxxxxxx")

audio_file= open("/content/audio.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
transcription=transcript.text
print(transcription)

# 4. Sentiment Analysis on text

# Need to add
!pip install -q transformers

from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["I like pizza"]
data=[transcription]
sentiment_pipeline(data)

text_sentiment = sentiment_pipeline(data)[0]["label"]

#text_sentiment = "Happy"

"""## Step 3:

LLM with some prompt engineering adjustments
"""

text = "hi"


prompt = f"You are a therapist and your client looks {emotion} while they are speaking with a {tone_emotion} tone and their words have a {text_sentiment} sentiment."

text = transcription

client = OpenAI(
    api_key="xxxxxxx",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role":"system",
            "content": prompt
        }
        {
            "role": "user",
            "content": text,
        }
    ],
    model="gpt-3.5-turbo",
)

thrapist_output = chat_completion.choices[0].message.content

print(chat_completion.choices[0].message.content)

"""## Step 4:

Text to speech
Nova and/or Alloy sounds best?

Potential user input for selecting voice?
"""

#thrapist_output = "Hi how are you?"
voice_input = "alloy"

client = OpenAI(api_key="xxxxxxx",)

response = client.audio.speech.create(
    model="tts-1",
    voice=voice_input,
    input=thrapist_output,
)

response.stream_to_file("output.mp3")

"""## extra
work in progress (integrating all AI's together as shown in flowchart

"""

from google.colab import drive
drive.mount('/content/drive')

# Setting a default path so its a bit easier to integrate into Flask later

import os

# Change this path to the path of your specific directory of the WAICY-2023 folder
default_path = '/content/drive/MyDrive/WAICY-2023'

os.chdir(default_path)

# We download every package that we need here:
!pip install transformers
!pip install openai
!pip install -r emotion_detection_cleaned/REQUIREMENTS.txt

from transformers import Wav2Vec2ForSequenceClassification, pipeline
import torchaudio
import torch
from openai import OpenAI
import subprocess
import soundfile as sf
import numpy as np


# Load the tone detection model
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Load sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

"""#Summary
WE NEED 4 PARTS

Transcription, LLM, Emotion detection, TTS

We also have tone detection and voice audio detection
"""

# UPDATE: 11/12/23

# Only need the following line:
!pip install openai

# Openai
# https://github.com/openai/openai-python/tree/main#installation

# To install new openai run
# !pip install openai==1

# To install old version run
# !pip install openai==0.28

"""#Tone detection"""

# Works
# python
# need wav file

!pip install transformers

from transformers import Wav2Vec2ForSequenceClassification
import torchaudio
import torch

# Load the model
model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

# Load and preprocess an audio file
audio_input, sample_rate = torchaudio.load("recordingsample.wav")

# Resample if necessary
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    audio_input = resampler(audio_input)

# Make sure the audio input is in the right shape: (batch_size, num_samples)
audio_input = audio_input.squeeze(0)  # Remove channel dimension if it exists
inputs = audio_input.unsqueeze(0)  # Add a batch dimension

# Predict emotion
with torch.no_grad():
    logits = model(inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# Output the predicted ID
print("Predicted Emotion ID:", predicted_ids.item())
emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
print("Predicted Emotion: ", emotions[predicted_ids.item()])
tone_emotion = emotions[predicted_ids.item()]

# https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition/tree/main

"""# Voice Audio Detection"""

!pip install webrtcvad
import webrtcvad
vad = webrtcvad.Vad()
vad.set_mode(3) # noise filterning, int 0 to 3 with 3 most agressive
sample_rate = 16000
frame_duration = 10  # ms, option of 10, 20, 30
frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
print('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))

"""#Transcription"""

### Works
# new openai
# python

from openai import OpenAI
client = OpenAI(api_key="xxxxxx")

audio_file= open("/content/output.mp3", "rb")
transcript = client.audio.transcriptions.create(
  model="whisper-1",
  file=audio_file
)
transcription=transcript.text
print(transcription)

# Other versions (depreciated)


### Working
# Old openai
# python

!pip install openai==0.28
import openai

openai.api_key = "xxxxx"

file = open("/content/recording_20231111_112212.mp3", "rb")
transcription = openai.Audio.transcribe("whisper-1", file)

print(transcription["text"])



### WORKING TRANSCRIPTION
# New openai
# curl = linux
# Need to transfer to python with subprocess

"""Working sample curl with openai==1


curl --request POST \
  --url https://api.openai.com/v1/audio/transcriptions \
  --header 'Authorization: Bearer xxxxxxx' \
  --header 'Content-Type: multipart/form-data' \
  --form file=@recording_20231111_112212.mp3\
  --form model=whisper-1

"""

# https://platform.openai.com/docs/guides/speech-to-text

# whisper
# https://github.com/openai/whisper
# slow
# not worth it

"""#llm"""

### Works working
# new version
# python

text = "hi"
text = transcription

from openai import OpenAI

client = OpenAI(
    api_key="xxxxx",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": text,
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content)

#Depreciated alternative versions

### Curl probably works for new version
# new version
# curl - linux
# need to subprocess it
"""
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer xxxxxxx" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Who won the world series in 2020?"
      },
      {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020."
      },
      {
        "role": "user",
        "content": "Where was it played?"
      }
    ]
  }'
"""


### WORKS
# old version
# Python

prompt = "Hello"
openai.api_key = 'xxxxx'

request = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    messages=[
        {"role": "user", "content": prompt},
    ]
)
message = request['choices'][0]['message']['content']
message

# https://platform.openai.com/docs/guides/text-generation

"""#TTS"""

### Works
# new version
# python
# audio streaming
from openai import OpenAI

client = OpenAI(api_key="xxxx",)

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="School is so boring. I hate the fact that I always have so much homework everyday!",
)

response.stream_to_file("output.mp3")

# Depreciated

### WORKS
# new version
# curl - linux
# need to put into subprocess

"""
curl https://api.openai.com/v1/audio/speech \
  -H "Authorization: Bearer xxxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Today is a wonderful day to build something people love!",
    "voice": "alloy"
  }' \
  --output speech.mp3
  """

### NOT TESTED
# audio streaming rather than file
# new version
# python

from pathlib import Path
from openai import OpenAI
client = OpenAI(api_key="xxxxxx",)

speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="alloy",
  input="Today is a wonderful day to build something people love!"
)

response.stream_to_file(speech_file_path)

#Echo and nova good voices
# https://platform.openai.com/docs/guides/text-to-speech

# TTS
# https://platform.openai.com/docs/guides/text-to-speech

"""#Emotion detection"""

#Check out github for local model

#GITHUB: https://github.com/AregGevorgyan/WAICY2023/tree/main/emotion_detection_cleaned

"""#Other notes"""

# what is async????

import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="xxxxx",
)


async def main() -> None:
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Say this is a test",
            }
        ],
        model="gpt-3.5-turbo",
    )


asyncio.run(main())


#https://github.com/openai/openai-python#async-usage

"""# Vision???"""

# Not tested...

from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])
