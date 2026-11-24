# Basic ML API

신뢰된 내부 Keras 모델 아카이브를 업로드하고, 필요할 때 메모리에 적재해 예측을 제공하는 Flask 서비스입니다. 모델 저장소 교체, LRU 캐시, 오래된 모델 정리, Prometheus 메트릭을 한 프로세스 안에서 관리합니다.

> [!WARNING]
> 이 서비스에는 인증·인가가 없습니다. Keras 모델 로딩은 샌드박스가 아니므로 신뢰할 수 있는 모델 생산자만 업로드해야 하며, 인증 계층 없이 공용 인터넷에 노출하면 안 됩니다.

## 빠른 시작

Python 3.11과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```bash
uv sync --frozen --group dev
uv run python examples/create_demo_model.py
uv run python -m src.main
```

별도 터미널에서 데모 모델을 업로드하고 예측합니다.

```bash
curl --fail \
  --form "model_file=@demo-model.zip" \
  "http://127.0.0.1:5000/upload_model?hash=demo-model-v1"

curl --fail \
  --header "Content-Type: application/json" \
  --data '[[3.0]]' \
  "http://127.0.0.1:5000/predict?hash=demo-model-v1"
```

예측 응답은 다음 형태입니다.

```json
{
  "data": {
    "model_hash": "demo-model-v1",
    "prediction": [[6.0]]
  }
}
```

## HTTP 인터페이스

| Method | Path | 입력 | 역할 |
|---|---|---|---|
| `POST` | `/upload_model?hash=...` | `multipart/form-data`, 필드 `model_file` | ZIP 안의 `.keras` 모델을 검증하고 원자적으로 설치 |
| `GET` | `/get_model?hash=...` | 쿼리 `hash` | 등록된 모델 경로와 마지막 사용 시각 조회 |
| `POST` | `/predict?hash=...` | JSON 배열 | 모델을 캐시에서 가져오거나 로드한 뒤 예측 |
| `GET` | `/health` | 없음 | 프로세스 생존 확인 |
| `GET` | `/metrics` | 없음 | Prometheus 텍스트 메트릭 |

모델 해시는 ASCII 문자·숫자·점·밑줄·하이픈만 허용하며 길이는 8~128자입니다. 오류 응답은 사람이 읽는 문구가 아니라 안정적인 `error.code`로 분기합니다.

```json
{
  "error": {
    "code": "model_not_found",
    "message": "The requested model was not found.",
    "details": {"model_hash": "demo-model-v1"}
  }
}
```

전체 계약은 [`docs/SPEC.md`](docs/SPEC.md), 상태·동시성 경계는 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)에 있습니다.

## 저장소 지도

```text
src/main.py                 Flask 조립 지점과 모니터링 시작
src/api/                    HTTP 파싱·응답·오류 변환
src/common/                 UTC 시계, 오류 타입, 메트릭
src/core/model_archive.py   ZIP 검증과 안전한 추출
src/core/model_manager.py   저장소, 캐시, 교체, 정리, 동시성
src/core/model_types.py     서비스 결과 타입과 JSON 변환
tests/                      공개 계약과 실패 복구 회귀 테스트
```

## 설정

| 환경 변수 | 기본값 | 설명 |
|---|---:|---|
| `MODEL_STORE_PATH` | `data/model_` | 추출한 모델 디렉터리 |
| `MAX_MODEL_FILE_SIZE` | `104857600` | 업로드 파일과 압축 해제 결과의 최대 바이트 수 |
| `MAX_CACHED_MODELS` | `10` | 프로세스별 LRU 모델 캐시 크기 |
| `MODEL_CLEANUP_INTERVAL` | `5` | 정리 주기(시간), `0`이면 비활성화 |
| `SERVER_HOST` | `0.0.0.0` | 개발 서버 바인딩 주소 |
| `SERVER_PORT` | `5000` | 개발 서버 포트 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |
| `LOKI_URL` | 미설정 | 선택적 Loki 엔드포인트 |
| `APP_NAME` | `ml-api` | 로그·메트릭 태그 |
| `APP_ENV` | `dev` | 실행 환경 태그 |

로컬 값은 `.env.example`을 참고합니다. 애플리케이션은 `.env`를 직접 읽지 않으므로 셸, 프로세스 관리자 또는 Docker Compose가 환경 변수를 주입해야 합니다.

## 품질 확인

```bash
uv run ruff format --check .
uv run ruff check .
uv run python -m pytest -q
```

변경 규칙과 완료 조건은 [`CONTRIBUTING.md`](CONTRIBUTING.md)에 있습니다.

## Docker

```bash
docker compose build
# 최초 실행 또는 0.3.1에서 업그레이드할 때만 실행합니다.
docker compose stop basic_ml_api
docker compose run --rm --no-deps -T --user 0:0 \
  --cap-add CHOWN --cap-add DAC_OVERRIDE --entrypoint sh basic_ml_api \
  -eu -c 'test ! -L /app/data/model_; mkdir -p /app/data/model_; chown -R --no-dereference 10001:10001 /app/data/model_'
docker compose up -d
docker compose exec -T basic_ml_api python -c \
  "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health').close()"
```

호스트에서는 기본적으로 `http://127.0.0.1:12021`로 접근합니다. 기존 버전과 동일한 호스트 경로 `./data/model_`에 모델을 보존합니다. 업그레이드 전 데이터를 백업하고, 위 초기화 명령으로 모델 디렉터리 소유권만 런타임 UID/GID(10001:10001)에 맞춥니다.

## 운영 제약

- 모델 메타데이터와 캐시는 프로세스 로컬 상태입니다. 기본 Gunicorn 설정은 **worker 1개**이며, 여러 worker 또는 여러 replica를 쓰려면 공유 메타데이터·캐시·락 설계가 먼저 필요합니다.
- 시작 시 모델 디렉터리를 다시 스캔하므로 마지막 사용 시각은 영속화되지 않습니다.
- 업로드 API는 인증을 제공하지 않습니다. 외부 노출 시 별도의 인증 프록시와 업로드 권한 통제가 필요합니다.
- 공식 Docker 이미지는 CPU 전용 런타임입니다(Linux x86-64: `tensorflow-cpu`, ARM64: `tensorflow`). GPU 실행은 별도 이미지와 운영 계약이 필요합니다.
- TensorFlow/Keras 모델은 신뢰 경계 안에서만 취급합니다. `safe_mode=True`는 샌드박스를 의미하지 않습니다.

## 라이선스

[MIT](LICENSE)
