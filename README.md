# webrtc-opencv
.tar 파일 2개 구글 드라이브에서 가져오기 

!! python 3.7에서만 작동 

1. pip install -r requirements.txt
2. pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

3. pip install flask 등등 오류 뜨는 모듈 전부 설치

4. set FLASK_APP=stream_imag.py
5. flask run --host=0.0.0.0
