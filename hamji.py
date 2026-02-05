import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from scipy.signal import butter, lfilter

# 1. 고음역대를 부드럽게 만드는 필터
def low_pass_filter(data, cutoff, sr, order=4):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# 2. 따뜻한 느낌을 주는 저음역대 강조 필터
def warm_boost_filter(data, sr):
    # 200Hz~500Hz 부근을 살짝 살려주면 목소리가 상냥하고 따뜻해집니다.
    b, a = butter(2, [200/(0.5*sr), 500/(0.5*sr)], btype='band')
    warm_part = lfilter(b, a, data)
    return data + (warm_part * 0.5)

st.set_page_config(page_title="보들보들 보이스 스튜디오 v3", page_icon="☁️")
st.title("☁️ 보들보들 상냥한 보이스 (최종 진화형)")

uploaded_file = st.file_uploader("녹음 파일을 올려주세요", type=['m4a', 'wav', 'mp3'])

if uploaded_file:
    with open("input.m4a", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("AI 노이즈 제거 및 보이스 클리닝 진행 중...")

    # [처리 1] 오디오 로드
    y, sr = librosa.load("input.m4a", sr=None)

    # [처리 2] 강력한 노이즈 제거 (AI 방식)
    # 주변의 '쉬-' 하는 소리나 화이트 노이즈를 마법처럼 지워줍니다.
    y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)

    # [처리 3] 피치 조절 (더 미세하게!)
    # 상냥한 느낌을 위해 0.8 ~ 1.2 사이를 추천합니다. (이번엔 1.0으로 더 자연스럽게!)
    y_pitched = librosa.effects.pitch_shift(y_denoised, sr=sr, n_steps=0.8)

    # [처리 4] 보들보들 필터링 (고음역대 컷)
    # 3500Hz로 더 낮춰서 아주 포근한 소리를 만듭니다.
    y_soft = low_pass_filter(y_pitched, cutoff=3500, sr=sr)

    # [처리 5] 따뜻함 추가 (저음 강조)
    y_warm = warm_boost_filter(y_soft, sr)

    # [처리 6] 속도 및 볼륨 최적화
    y_final = librosa.effects.time_stretch(y_warm, rate=0.96) # 조금 더 천천히
    
    # 볼륨 정규화 (찢어지는 소리 방지)
    max_val = np.max(np.abs(y_final))
    if max_val > 0:
        y_final = y_final / max_val * 0.6

    # 결과 저장
    output_path = "final_soft_voice.wav"
    sf.write(output_path, y_final, sr)

    st.success("이제 훨씬 깨끗하고 부드러운 목소리가 되었을 거예요!")
    st.audio(output_path)
    
    with open(output_path, "rb") as f:
        st.download_button("상냥한 목소리 다운로드", f, file_name="final_voice.wav")
