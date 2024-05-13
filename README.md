# basic_ml_api
이 프로젝트는 Flask 기반 웹 서비스로, 모델을 캐싱하고 관리하며 데이터에 대한 예측을 수행하는 것을 목적으로 합니다. 사용자는 모델 파일을 업로드하고 관리할 수 있으며, Prometheus를 사용하여 서비스 메트릭을 모니터링할 수 있습니다. 이 웹 서비스는 머신러닝 모델의 효율적인 배포와 모니터링을 위해 설계되었습니다.

## 기능
- 모델 파일 업로드 및 추출
- 업로드된 모델의 메타데이터 관리
- 캐싱 시스템을 통한 모델 로드 및 관리
- 모델을 사용한 데이터에 대한 예측 수행
- Prometheus를 통한 서비스 메트릭 모니터링
- 오래된 모델 자동 삭제

## 시작하기

### 전제 조건
이 프로젝트는 Python 3.10 기반으로 작성되었습니다.

### 설치
이 저장소를 클론합니다.
```bash
git clone https://github.com/humaningansalam/basic_ml_api.git
```

## 필요한 패키지를 설치합니다.
```bash
pip install -r requirements.txt
```

## 사용법
서버를 시작합니다.
```bash
python main.py
```

# 서비스 상호작용 엔드포인트

- 모델 업로드: `POST /upload_model`
- 모델 존재 확인: `GET /get_model`
- 예측 수행: `POST /predict`
- 서비스 상태 확인: `GET /health`

## API 참조

### `POST /upload_model`
모델 파일과 그 해시를 받아서 저장하고 압축 파일을 추출합니다.

#### 요청 파라미터
- `model_file`: (파일) 업로드할 모델 파일 (.zip 형식)
- `hash`: (쿼리 문자열) 모델 파일의 해시 값

### `GET /get_model`
주어진 해시 값의 모델 메타데이터를 반환합니다.

#### 요청 파라미터
- `hash`: (쿼리 문자열) 조회할 모델의 해시 값

### `POST /predict`
지정된 모델을 사용하여 주어진 데이터에 대한 예측을 수행합니다.

#### 요청 파라미터
- `data`: (JSON 본문) 예측에 사용될 데이터
- `hash`: (쿼리 문자열) 사용될 모델의 해시 값

### `GET /health`
서비스의 상태를 확인합니다. 서비스가 정상적으로 실행 중이면 "Healthy"를 반환합니다.

## Prometheus 메트릭
- `app_cpu_usage`: cpu 사용량
- `app_ram_usage`: ram 사용량
- `model_cache_usage`: 캐시된 모델의 수
- `errors`: 발생한 오류의 수
- `predictions_completed`: 완료된 예측의 수

## 유지 관리
오래된 모델은 자동으로 삭제됩니다. 사용기간이 일주일 이상 된 모델을 청소하기 위해 `del_oldmodel` 함수를 사용할 수 있습니다.


## 라이센스
이 프로젝트는 MIT License 하에 배포됩니다.
