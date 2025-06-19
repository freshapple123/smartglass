from gtts import gTTS
import io
import pygame  # 음성 재생용

"""
pip install gTTS pygame
"""

# 텍스트를 TTS로 변환
text = "안녕하세요. 파일 없이 바로 들려드릴게요."
tts = gTTS(text=text, lang="ko")

# mp3를 메모리 버퍼에 저장
fp = io.BytesIO()
tts.write_to_fp(fp)
fp.seek(0)

# pygame으로 mp3 재생
pygame.mixer.init()
pygame.mixer.music.load(fp)
pygame.mixer.music.play()

# 재생이 끝날 때까지 대기
while pygame.mixer.music.get_busy():
    continue
