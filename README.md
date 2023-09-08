# webrtc-opencv

<a href="https://github.com/checkmate2022/Backend/wiki"><img width="240" alt="image" src="https://user-images.githubusercontent.com/62784314/197489630-3230add5-241b-4fa6-9282-ecd4811c1420.png"></a>  



.tar 파일 2개 구글 드라이브에서 가져오기 

!! python 3.7에서만 작동 

1. pip install -r requirements.txt
2. pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

3. pip install flask 등등 오류 뜨는 모듈 전부 설치

4. set FLASK_APP=stream_imag.py
5. flask run --host=0.0.0.0




웹소켓
1. redis-cli 설치 https://velog.io/@6v6/%EC%9C%88%EB%8F%84%EC%9A%B0%EC%97%90-Redis-%EC%84%A4%EC%B9%98%ED%95%98%EA%B3%A0-Redis-cli%EB%A1%9C-%EC%A1%B0%ED%9A%8C%ED%95%98%EA%B8%B0
2. requirements.txt 추가한 부분 설치
3. python recorder.py
4. python server.py
5. http://127.0.0.1:9000/
