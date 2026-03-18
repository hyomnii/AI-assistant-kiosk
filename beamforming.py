import queue
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd
import whisper

# -----------------------------
# 최소 설정
# -----------------------------
RATE = 16000          # Whisper 권장 16k
CHANNELS = 2          # 듀얼 마이크(스테레오)
CHUNK = 1024
DEVICE_INDEX = None   # None=기본 입력장치(사운드 장치 인덱스)
MODEL_SIZE = "base"

# A: 고정 길이 전사
SEGMENT_SECONDS = 4.0

# B: 빔포밍 + 임계치/침묵 기반 전사
THRESHOLD_DB = -35.0
MIC_DISTANCE_M = 0.08
STEER_ANGLE_DEG = 0.0
SPEED_OF_SOUND = 343.0
SILENCE_TIMEOUT_SEC = 0.7
MIN_UTTERANCE_SEC = 0.6
MAX_UTTERANCE_SEC = 8.0

# 상태 출력(원하면 False)
PRINT_LEVEL = True
PRINT_INTERVAL_SEC = 1.0


def rms_db(x: np.ndarray) -> float:
    """RMS 기반 dBFS 계산(x는 -1~1 float 권장)."""
    eps = 1e-12
    x = x.astype(np.float32, copy=False)
    rms = np.sqrt(np.mean(np.square(x))) + eps
    return 20.0 * np.log10(rms)


def fractional_delay(signal: np.ndarray, delay_samples: float) -> np.ndarray:
    """
    분수 샘플 지연(1차 보간).
    - (+)면 뒤로 지연, (-)면 앞으로 당김
    """
    if abs(delay_samples) < 1e-6:
        return signal.astype(np.float32, copy=True)

    n = np.arange(len(signal), dtype=np.float32)
    src_pos = n - delay_samples
    src_pos = np.clip(src_pos, 0, len(signal) - 1.000001)

    i0 = np.floor(src_pos).astype(np.int32)
    frac = src_pos - i0
    i1 = np.clip(i0 + 1, 0, len(signal) - 1)

    sig = signal.astype(np.float32, copy=False)
    y = (1.0 - frac) * sig[i0] + frac * sig[i1]
    return y.astype(np.float32, copy=False)


def delay_and_sum_beamform(stereo: np.ndarray, rate: int, d_m: float, steer_deg: float) -> np.ndarray:
    """
    두 마이크 Delay-and-Sum 빔포밍(간단 버전).
    - stereo: (N, 2)
    - τ = (d * sin(θ)) / c, 각 채널을 ±τ/2 정렬 후 평균
    """
    theta = np.deg2rad(steer_deg)
    tdoa_sec = (d_m * np.sin(theta)) / SPEED_OF_SOUND
    tdoa_samples = tdoa_sec * rate

    delay_left = -tdoa_samples / 2.0
    delay_right = +tdoa_samples / 2.0

    left = stereo[:, 0]
    right = stereo[:, 1]

    left_aligned = fractional_delay(left, delay_left)
    right_aligned = fractional_delay(right, delay_right)

    y = 0.5 * (left_aligned + right_aligned)
    return y.astype(np.float32, copy=False)


def stereo_to_mono_avg(stereo: np.ndarray) -> np.ndarray:
    """스테레오 -> 모노(단순 평균)."""
    return np.mean(stereo, axis=1).astype(np.float32, copy=False)


def transcribe_loop(model, job_q: queue.Queue) -> None:
    """큐에서 오디오를 꺼내 전사해서 출력(백그라운드 스레드)."""
    while True:
        mode_label, audio = job_q.get()
        try:
            result = model.transcribe(audio, language="ko", fp16=False)
            text = (result.get("text") or "").strip()
            print(f"[{mode_label}] {text if text else '(빈 결과)'}")
        except Exception as e:
            print(f"[{mode_label}] 오류: {e}")
        finally:
            job_q.task_done()


def main() -> None:
    print("듀얼 마이크 STT (A=고정길이, B=빔포밍+침묵구간)")
    print("Whisper 모델 로딩...")
    model = whisper.load_model(MODEL_SIZE)
    print(f"모델: {MODEL_SIZE}")

    job_q = queue.Queue(maxsize=8)
    threading.Thread(target=transcribe_loop, args=(model, job_q), daemon=True).start()

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=32)

    def on_audio(indata: np.ndarray, frames: int, time_info, status) -> None:
        # 콜백은 최대한 가볍게(연산/전사 금지). 블록을 큐에 넣고 메인 루프에서 처리.
        if status:
            # 드롭/오버플로우 등이 보이면 참고용으로만 출력
            print(f"(오디오 상태) {status}")
        try:
            audio_q.put_nowait(indata.copy())
        except queue.Full:
            # 처리 속도가 입력을 못 따라가면 가장 먼저 버립니다(지연 누적 방지)
            pass

    max_samples = int(RATE * SEGMENT_SECONDS)
    buf_a = deque(maxlen=max_samples)
    buf_b = deque(maxlen=int(RATE * MAX_UTTERANCE_SEC))

    last_print = 0.0
    b_speaking = False
    b_last_loud_time = 0.0
    b_start_time = 0.0

    print("\n녹음을 시작합니다. Ctrl+C로 종료하세요.\n")
    try:
        with sd.InputStream(
            samplerate=RATE,
            channels=CHANNELS,
            blocksize=CHUNK,
            dtype="float32",
            device=DEVICE_INDEX,
            callback=on_audio,
        ):
            while True:
                stereo = audio_q.get()  # (CHUNK, CHANNELS) float32, -1~1
                if stereo.ndim != 2 or stereo.shape[1] != CHANNELS:
                    continue

                # 모드 A: 스테레오 평균 -> 고정 길이 전사
                mono_a = stereo_to_mono_avg(stereo)
                buf_a.extend(mono_a.tolist())

                # 모드 B: 빔포밍 -> 임계치 기반으로 말 구간만 모으기
                mono_beam = delay_and_sum_beamform(stereo, RATE, MIC_DISTANCE_M, STEER_ANGLE_DEG)
                current_db = rms_db(mono_beam)
                now = time.time()

                if current_db >= THRESHOLD_DB:
                    if not b_speaking:
                        b_speaking = True
                        b_start_time = now
                    b_last_loud_time = now
                    buf_b.extend(mono_beam.tolist())
                else:
                    if b_speaking and (now - b_last_loud_time) >= SILENCE_TIMEOUT_SEC:
                        utter_len = len(buf_b) / RATE
                        if utter_len >= MIN_UTTERANCE_SEC and (not job_q.full()):
                            seg_b = np.array(buf_b, dtype=np.float32)
                            job_q.put(("B", seg_b))
                        buf_b.clear()
                        b_speaking = False
                        b_last_loud_time = 0.0
                        b_start_time = 0.0

                if b_speaking and (now - b_start_time) >= MAX_UTTERANCE_SEC:
                    utter_len = len(buf_b) / RATE
                    if utter_len >= MIN_UTTERANCE_SEC and (not job_q.full()):
                        seg_b = np.array(buf_b, dtype=np.float32)
                        job_q.put(("B", seg_b))
                    buf_b.clear()
                    b_speaking = False
                    b_last_loud_time = 0.0
                    b_start_time = 0.0

                if PRINT_LEVEL and (now - last_print) >= PRINT_INTERVAL_SEC:
                    b_state = "말하는 중" if b_speaking else "침묵"
                    print(
                        f"B레벨 {current_db:6.1f} dBFS | B {b_state} | "
                        f"A {len(buf_a)}/{max_samples} | B {len(buf_b)/RATE:.2f}초"
                    )
                    last_print = now

                if len(buf_a) >= max_samples and not job_q.full():
                    seg_a = np.array(buf_a, dtype=np.float32)
                    job_q.put(("A", seg_a))
                    buf_a.clear()

    except KeyboardInterrupt:
        print("\n사용자 중지 요청. 정리 중...")
    finally:
        print("종료되었습니다.")


if __name__ == "__main__":
    main()
