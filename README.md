# basic_ml_api
이 프로젝트는 Flask 기반 웹 서비스로, 모델을 캐싱하고 관리하며 데이터에 대한 예측을 수행하는 것을 목적으로 합니다. 사용자는 모델 파일을 업로드하고 관리할 수 있으며, Prometheus를 사용하여 서비스 메트릭을 모니터링할 수 있습니다. 이 웹 서비스는 머신러닝 모델의 효율적인 배포와 모니터링을 위해 설계되었습니다.

## 기능
- **모델 관리**: 모델 파일 업로드, 압축 해제 및 메타데이터 관리
- **스마트 캐싱**: LRU 알고리즘을 통한 메모리 효율적 모델 로딩
- **예측 서빙**: 업로드된 모델을 사용한 실시간 예측
- **모니터링**: Prometheus 호환 리소스(CPU, RAM, Cache) 메트릭 제공
- **자동 정리**: 오랫동안 사용되지 않은 모델 자동 삭제

## 빠른 시작

### 전제 조건
이 프로젝트는 Python 3.11 기반으로 작성되었습니다.

### 1. 설치

```bash
git clone https://github.com/humaningansalam/basic_ml_api.git
cd basic_ml_api
uv sync
```

### 2. 데모 모델 생성

아래 스크립트는 입력값에 `2`를 곱하는 실제 Keras 모델을 `demo-model.zip`으로 생성합니다.

```bash
uv run python examples/create_demo_model.py
# Created /path/to/basic_ml_api/demo-model.zip
```

### 3. 서버 시작

```bash
uv run python -m src.main
```

다음 명령은 별도 터미널에서 실행합니다.

### 4. 업로드 및 예측

```bash
curl --fail \
  --form "model_file=@demo-model.zip" \
  "http://127.0.0.1:5000/upload_model?hash=demo-model-v1"
# {"message":"Model uploaded successfully"}

curl --fail \
  "http://127.0.0.1:5000/get_model?hash=demo-model-v1"
# {"message":{"file_path":".../demo-model-v1","used":"..."}}

curl --fail \
  --header "Content-Type: application/json" \
  --data '[[3.0]]' \
  "http://127.0.0.1:5000/predict?hash=demo-model-v1"
# {"prediction":[[6.0]]}
```

### 5. 운영 엔드포인트 확인

```bash
curl --fail http://127.0.0.1:5000/health
# Healthy

curl --fail http://127.0.0.1:5000/metrics
```

### Docker Compose
Docker Compose defaults to `VERSION=dev`, so a fresh checkout can render configuration without a local `.env` file.
If you want to override the image tag or app settings locally, add a `.env` file in `repo/`.
```bash
docker compose config
```

## API 참조

- **모델 업로드**: `POST /upload_model`
  - Body: `model_file` (.zip 파일)
  - Query: `hash` (모델 해시값)
- **모델 정보 조회**: `GET /get_model`
  - Query: `hash`
- **예측 수행**: `POST /predict`
  - Body: `data` (JSON 배열)
  - Query: `hash`
- **상태 확인**: `GET /health`
- **메트릭 조회**: `GET /metrics`

## Prometheus 메트릭
- `ml_api_cpu_usage_percent`: CPU 사용량 (%)
- `ml_api_ram_usage_mb`: RAM 사용량 (MB)
- `model_cache_usage`: 현재 캐시된 모델 수
- `cache_hits` / `cache_misses`: 캐시 적중/미적중 횟수
- `predictions_completed`: 예측 완료 횟수
- `ml_api_errors_total`: 에러 발생 횟수


## 라이센스
이 프로젝트는 MIT License 하에 배포됩니다.
