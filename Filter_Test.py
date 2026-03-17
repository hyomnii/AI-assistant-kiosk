import sounddevice as sd
import numpy as np
import whisper
import wave
import noisereduce as nr  # 노이즈 제거 라이브러리

# 1. Whisper 모델 로드 (성능을 위해 large 사용, 사양에 따라 base/medium 변경 가능)
print("Loading Whisper model...")
model = whisper.load_model("large")

# 2. 음성 녹음 및 전처리 함수
def record_and_process(filename="processed_audio1.wav", duration=5, fs=44100):
    print(f"Recording for {duration} seconds...")
    
    # 1채널(Mono)로 녹음
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")

    # [노이즈 제거 알고리즘 적용]
    # 마이크 1개 환경에서 정면 목소리를 부각하기 위해 주변 백색 소음을 억제합니다.
    reduced_noise_audio = nr.reduce_noise(y=audio_data.flatten(), sr=fs)

    # 데이터를 다시 16-bit PCM 형태로 변환 (WAV 저장용)
    audio_int16 = (reduced_noise_audio * 32767).astype(np.int16)

    # 3. 파일 저장
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_int16.tobytes())
    
    return filename

# 4. 텍스트 변환 함수 (Whisper 최적화)
def transcribe_audio(filename):
    print("Transcribing...")
    # language="ko"를 지정하면 정면 목소리 판별력이 높아집니다.
    result = model.transcribe(filename, language="ko", fp16=False)
    print("-" * 30)
    print("Transcription Result:")
    print(result["text"])
    print("-" * 30)

# 메인 실행부
if __name__ == "__main__":
    # 녹음 및 잡음 제거 수행
    audio_file = record_and_process(duration=5)
    
    # 결과 확인 및 텍스트 변환
    transcribe_audio(audio_file)