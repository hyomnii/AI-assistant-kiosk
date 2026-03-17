import sounddevice as sd
import numpy as np
import whisper
import wave

# Whisper 모델 로드
model = whisper.load_model("large")

# 음성 녹음 함수
def record_audio(filename="recorded_audio.wav", duration=5, fs=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # 녹음이 끝날 때까지 대기
    print("Recording finished.")
    
    # 녹음된 데이터를 WAV 파일로 저장
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit sample width
        wf.setframerate(fs)
        wf.writeframes(audio_data.tobytes())

# 텍스트로 변환하는 함수
def transcribe_audio(filename="recorded_audio.wav"):
    result = model.transcribe(filename)
    print("Transcription: ", result["text"])

# 메인 실행
if __name__ == "__main__":
    record_audio("recorded_audio.wav", duration=5)  # 5초 동안 음성 녹음
    transcribe_audio("recorded_audio.wav")  # 녹음된 오디오를 텍스트로 변환
