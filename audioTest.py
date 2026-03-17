import soundfile as sf
import numpy as np
import sounddevice as sd

# 오디오 파일 읽기
audio_file = r'C:\Users\joh82\Documents\GitHub\AI-assistant-kiosk\recorded_audio.wav'
audio_data, sample_rate = sf.read(audio_file)

# 오디오 데이터 재생
sd.play(audio_data, sample_rate)
sd.wait()  # 오디오가 끝날 때까지 기다림