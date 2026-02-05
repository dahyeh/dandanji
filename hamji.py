import streamlit as st
import librosa
import soundfile as sf
import numpy as np

# 웹사이트 디자인
st.set_page_config(page_title="부드러운 보이스 스튜디오", page_icon="🎙️")
st.title("🎙️ 상냥하고 조곤조곤한 목소리 변환기")
st.write("아이폰 m4a 파일을 업로드하면, 다정한 라디오 톤으로 바꿔드려요.")

# 비밀번호 보안 (원하는 대로 수정하세요)
password = st.sidebar.text_input("접속 비밀번호", type="password")
if password != "1234":
    st.warning("비밀번호를 입력해 주세요.")
    st.stop()

# 파일 업로드
uploaded_file = st.file_uploader("녹음 파일을 선택하세요", type=['m4a', 'mp3', 'wav'])

if uploaded_file:
    # 1. 파일 임시 저장
    with open("input.m4a", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info("부드러운 목소리로 다듬는 중입니다... ✨")

    # 2. 오디오 로드
    y, sr = librosa.load("input.m4a", sr=None)

    # [마법 설정 1] 피치 살짝 올리기 (상냥한 톤)
    # 1.5는 너무 튀지 않으면서도 목소리가 맑아지는 수치입니다.
    y_gentle = librosa.effects.pitch_shift(y, sr=sr, n_steps=1.5)

    # [마법 설정 2] 속도 살짝 늦추기 (조곤조곤한 느낌)
    # 0.95는 아주 미세하게 천천히 말하게 하여 여유를 줍니다.
    y_calm = librosa.effects.time_stretch(y_gentle, rate=1.1)

    # [마법 설정 3] 음량 고르게 만들기 (부드러운 느낌)
    y_final = librosa.util.normalize(y_calm)

    # 3. 결과 저장 및 출력
    output_path = "gentle_voice.wav"
    sf.write(output_path, y_final, sr)

    st.success("변환이 완료되었습니다!")
    st.audio(output_path)
    
    with open(output_path, "rb") as f:
        st.download_button("상냥한 목소리 저장하기 (.wav)", f, file_name="soft_voice.wav")