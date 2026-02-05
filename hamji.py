import streamlit as st
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter

# --- 엔지니어의 비밀 도구: 고음 깎기(LPF) 함수 ---
def low_pass_filter(data, cutoff, sr, order=5):
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

st.set_page_config(page_title="프리미엄 보이스 스튜디오", page_icon="✨")
st.title("✨ 보들보들 상냥한 보이스 필터")

# 비밀번호 보안
password = st.sidebar.text_input("접속 비밀번호", type="password")
if password != "1234": # 본인이 설정한 비밀번호로 바꾸세요!
    st.warning("비밀번호를 입력해 주세요.")
    st.stop()

uploaded_file = st.file_uploader("녹음 파일을 올려주세요", type=['m4a', 'wav', 'mp3'])

if uploaded_file:
    with open("input.m4a", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("상냥한 목소리로 정밀 튜닝 중입니다... 🎙️")

    # 1. 파일 불러오기
    y, sr = librosa.load("input.m4a", sr=None)

    # 2. 목소리가 없는 부분의 잡음 제거 (Gate)
    yt, _ = librosa.effects.trim(y, top_db=25) 

    # 3. 피치 조절 (너무 높지 않게 1.2로 설정 - 성인 여성의 맑은 톤)
    y_pitched = librosa.effects.pitch_shift(yt, sr=sr, n_steps=1.2)

    # 4. 보들보들하게 만들기 (4000Hz 이상의 날카로운 소리 제거)
    y_smooth = low_pass_filter(y_pitched, cutoff=4000, sr=sr)

    # 5. 조곤조곤하게 속도 조절 (0.97배로 살짝 여유 있게)
    y_final = librosa.effects.time_stretch(y_smooth, rate=0.97)

    # 6. 소리가 깨지지 않게 볼륨 조절 (Normalization)
    max_val = np.max(np.abs(y_final))
    if max_val > 0:
        y_final = y_final * (0.7 / max_val)

    # 결과 저장
    output_path = "pro_soft_voice.wav"
    sf.write(output_path, y_final, sr)

    st.success("완료되었습니다! 훨씬 듣기 편해졌을 거예요.")
    st.audio(output_path)
    
    with open(output_path, "rb") as f:
        st.download_button("상냥한 목소리 저장하기", f, file_name="soft_voice.wav")
