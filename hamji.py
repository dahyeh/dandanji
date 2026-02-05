import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter, iirnotch

# --- [도구 1] 동굴 소리 제거 ---
def anti_boxness_filter(data, sr):
    f0 = 400.0
    Q = 0.7
    b, a = iirnotch(f0, Q, sr)
    return lfilter(b, a, data)

# --- [도구 2] 하이톤 '치지지' 노이즈 제거 (Smoothing) ---
def de_hiss_filter(data, sr, cutoff=6000):
    # 날카로운 고음역대를 부드럽게 깎아 실크 같은 질감을 만듭니다.
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

st.set_page_config(page_title="실크 보이스 스튜디오", page_icon="✨")
st.title("✨ '치지지' 소리 없는 매끄러운 상냥 보이스")

uploaded_file = st.file_uploader("녹음 파일을 올려주세요", type=['m4a', 'wav', 'mp3'])

if uploaded_file:
    with open("input.m4a", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("하이톤 노이즈를 매끄럽게 다듬는 중...")

    # [1] 로드 및 AI 노이즈 제거 (강도 하향 조정으로 음질 보존)
    y, sr = librosa.load("input.m4a", sr=None)
    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7) # 0.85 -> 0.7

    # [2] 동굴 소리 제거
    y_no_box = anti_boxness_filter(y_denoised, sr)

    # [3] 피치 조절
    y_pitched = librosa.effects.pitch_shift(y_no_box, sr=sr, n_steps=0.9)

    # [4] 하이톤 '치지지' 노이즈 제거 (De-hissing)
    # 6000Hz 이상의 날카로운 성분을 부드럽게 만듭니다.
    y_smooth = de_hiss_filter(y_pitched, sr, cutoff=6000)
    
    # [5] 마지막 다듬기 (조곤조곤한 속도)
    y_final = librosa.effects.time_stretch(y_smooth, rate=1.0) # 속도를 1.0으로 더 자연스럽게
    
    # 볼륨 최적화 (0.7로 낮춰서 피크 왜곡 방지)
    max_val = np.max(np.abs(y_final))
    if max_val > 0:
        y_final = y_final / max_val * 0.7

    output_path = "silk_voice.wav"
    sf.write(output_path, y_final, sr)

    st.success("이제 소리가 훨씬 매끄럽고 포근해졌을 거예요!")
    st.audio(output_path)
    
    with open(output_path, "rb") as f:
        st.download_button("매끄러운 목소리 저장", f, file_name="silk_voice.wav")
