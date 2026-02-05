import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter, iirnotch

# --- [엔지니어링 도구 1] 동굴 소리(웅웅거림) 제거 필터 ---
def anti_boxness_filter(data, sr):
    # 400Hz 부근의 '웅웅'거리는 주파수를 찾아 정밀하게 깎아냅니다.
    f0 = 400.0  # 타겟 주파수
    Q = 0.7     # 폭 (숫자가 낮을수록 부드럽게 깎임)
    b, a = iirnotch(f0, Q, sr)
    return lfilter(b, a, data)

# --- [엔지니어링 도구 2] 고음역대 보정 (먹먹함 방지) ---
def high_shelf_filter(data, sr, gain=3):
    # 너무 먹먹해지지 않게 고음역대를 살짝만 살려줍니다.
    nyq = 0.5 * sr
    cutoff = 3000 / nyq
    b, a = butter(2, cutoff, btype='high')
    high_part = lfilter(b, a, data)
    return data + (high_part * 0.2)

st.set_page_config(page_title="클로즈업 보이스 스튜디오", page_icon="🎙️")
st.title("🎙️ 동굴 소리 없는 '밀착형' 상냥 보이스")

uploaded_file = st.file_uploader("녹음 파일을 올려주세요", type=['m4a', 'wav', 'mp3'])

if uploaded_file:
    with open("input.m4a", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("동굴 울림을 제거하고 목소리를 앞으로 당기는 중...")

    # [1] 로드 및 AI 노이즈 제거
    y, sr = librosa.load("input.m4a", sr=None)
    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.85)

    # [2] 동굴 소리(웅웅거림) 정밀 제거 (핵심!)
    y_no_box = anti_boxness_filter(y_denoised, sr)

    # [3] 피치 조절 (가장 자연스러운 0.9단계)
    y_pitched = librosa.effects.pitch_shift(y_no_box, sr=sr, n_steps=0.9)

    # [4] 음색 보정 (고음은 살리고 지저분한 초고음만 컷)
    y_clear = high_shelf_filter(y_pitched, sr)
    
    # [5] 마지막 다듬기 (조곤조곤한 속도)
    y_final = librosa.effects.time_stretch(y_clear, rate=1.1)
    
    # 볼륨 최적화 및 리미팅 (소리가 깨지지 않게)
    y_final = np.clip(y_final, -1.0, 1.0)
    max_val = np.max(np.abs(y_final))
    if max_val > 0:
        y_final = y_final / max_val * 0.8

    output_path = "final_studio_voice.wav"
    sf.write(output_path, y_final, sr)

    st.success("이제 훨씬 선명하고 바로 옆에서 말하는 것 같을 거예요!")
    st.audio(output_path)
    
    with open(output_path, "rb") as f:
        st.download_button("최종 결과물 저장", f, file_name="studio_voice.wav")
