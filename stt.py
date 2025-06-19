import speech_recognition as sr

"""
pip install SpeechRecognition
pip install pyaudio  # 마이크 입력용
"""

# 마이크에서 음성 입력을 받기 위한 Recognizer 객체 생성
recognizer = sr.Recognizer()

print("🎤 말해주세요 (녹음 중)...")

# 마이크 열기
with sr.Microphone() as source:
    # 잡음 제거 (환경 적응)
    recognizer.adjust_for_ambient_noise(source)
    print("듣는 중...")
    audio = recognizer.listen(source)

    try:
        # 구글 웹 API를 이용한 음성 인식
        text = recognizer.recognize_google(audio, language="ko-KR")
        print("📝 인식된 내용:", text)

    except sr.UnknownValueError:
        print("❌ 음성을 인식하지 못했습니다.")
    except sr.RequestError as e:
        print(f"❌ 요청 실패: {e}")
