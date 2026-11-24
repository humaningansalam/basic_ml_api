# Basic ML API Specification

이 문서는 외부 클라이언트와 운영자가 의존할 수 있는 현재 계약을 정의합니다.

## 공통 응답

성공하는 JSON API는 최상위 `data`를 사용합니다.

```json
{"data": {}}
```

오류는 다음 envelope를 사용합니다.

```json
{
  "error": {
    "code": "model_not_found",
    "message": "The requested model was not found.",
    "details": {"model_hash": "demo-model-v1"}
  }
}
```

클라이언트는 변경 가능한 `message`가 아니라 `code`로 분기해야 합니다. `details`는 항상 객체입니다.

## 모델 해시

`hash` 쿼리 값은 다음 정규식을 만족해야 합니다.

```text
^[A-Za-z0-9._-]{8,128}$
```

해시는 저장 디렉터리 이름과 동시성 단위로 사용됩니다.

## 엔드포인트

### `POST /upload_model?hash={model_hash}`

요청:

- Content-Type: `multipart/form-data`
- 파일 필드: `model_file`
- 형식: 최소 한 개의 일반 `.keras` 파일을 포함한 ZIP

성공 `200`:

```json
{
  "data": {
    "model_hash": "demo-model-v1",
    "replaced": false
  }
}
```

`replaced`는 같은 해시의 기존 모델 디렉터리가 실제로 교체됐는지를 나타냅니다.

검증 규칙:

- 업로드 파일은 `MAX_MODEL_FILE_SIZE`를 넘을 수 없습니다.
- ZIP 엔트리는 최대 5,000개입니다.
- 압축 해제 크기의 합은 `MAX_MODEL_FILE_SIZE`를 넘을 수 없습니다.
- 절대 경로, `..`, 역슬래시 경로, 심볼릭 링크, 파일/디렉터리 충돌을 거부합니다.
- 검증과 추출은 최종 모델 디렉터리 밖의 격리된 작업공간에서 수행합니다.
- 추출한 Keras 모델은 예측과 동일한 로더로 검증한 후에만 설치합니다.
- 기존 모델은 백업 후 원자적으로 교체하며 설치 실패 시 복구합니다.

주요 오류:

| Status | Code | 조건 |
|---:|---|---|
| 400 | `model_file_required` | 파일 필드가 없거나 파일명이 비어 있음 |
| 400 | `model_hash_required` | 해시 누락 |
| 400 | `invalid_model_hash` | 허용하지 않는 해시 형식 |
| 413 | `upload_too_large` | 업로드 파일 한도 초과 |
| 400 | `invalid_zip` | ZIP 형식, 압축 데이터 또는 엔트리 구조가 잘못됨 |
| 400 | `zip_entry_limit_exceeded` | 엔트리 수 초과 |
| 400 | `unsafe_zip_entry` | 안전하지 않은 경로 또는 링크 |
| 400 | `uncompressed_size_exceeded` | 압축 해제 크기 한도 초과 |
| 400 | `model_artifact_required` | `.keras` 파일 없음 |
| 400 | `invalid_model_artifact` | Keras 모델을 로드할 수 없음; 기존 모델 유지 |
| 500 | `model_storage_failed` | 파일 저장, 이동, 삭제 실패 |

### `GET /get_model?hash={model_hash}`

성공 `200`:

```json
{
  "data": {
    "model_hash": "demo-model-v1",
    "file_path": "data/model_/demo-model-v1",
    "used": "2026-11-24T06:00:00+00:00"
  }
}
```

`used`는 timezone-aware ISO 8601 UTC 시각입니다.

오류:

| Status | Code | 조건 |
|---:|---|---|
| 400 | `model_hash_required` | 해시 누락 |
| 404 | `model_not_found` | 등록되지 않은 해시 |

### `POST /predict?hash={model_hash}`

요청:

- Content-Type: `application/json`
- 본문: 비어 있지 않은 JSON 배열
- `NaN`, `Infinity`, `-Infinity` 같은 비표준 JSON 숫자는 허용하지 않음

성공 `200`:

```json
{
  "data": {
    "model_hash": "demo-model-v1",
    "prediction": [[6.0]]
  }
}
```

NumPy 배열, TensorFlow tensor, 목록, tuple, 문자열 키를 가진 객체의 중첩 구조를 보존해 JSON 값으로 변환합니다. 비유한 부동소수점 값과 지원하지 않는 출력 타입은 실패합니다.

오류:

| Status | Code | 조건 |
|---:|---|---|
| 400 | `model_hash_required` | 해시 누락 |
| 400 | `prediction_data_required` | JSON 본문 누락 |
| 400 | `malformed_json` | 잘못된 JSON 또는 비표준 숫자 |
| 400 | `prediction_data_not_array` | 최상위 값이 배열이 아님 |
| 400 | `prediction_data_empty` | 빈 배열 |
| 400 | `invalid_prediction_data` | ragged 배열, 과도한 중첩 등 변환 불가 |
| 404 | `model_not_found` | 등록되지 않은 모델 |
| 409 | `model_state_conflict` | 로드 중 모델 경로가 교체됨 |
| 500 | `model_artifact_unavailable` | 등록 경로에 `.keras` 파일이 없음 |
| 500 | `prediction_failed` | 모델 로드, 실행 또는 결과 변환 실패 |

### `GET /health`

성공 `200`:

```json
{"data": {"status": "healthy"}}
```

이 엔드포인트는 외부 의존성의 준비 상태가 아니라 현재 프로세스가 요청에 응답하는지만 확인합니다.

### `GET /metrics`

성공 `200`, Prometheus text exposition 형식입니다.

주요 메트릭:

- `model_cache_usage`
- `predictions_completed_total`
- `cache_hits_total`
- `cache_misses_total`
- `ml_api_errors_total{type="..."}`
- `ml_api_cpu_usage_percent`
- `ml_api_ram_usage_mb`

## 프레임워크 오류

| Status | Code | 조건 |
|---:|---|---|
| 404 | `route_not_found` | 존재하지 않는 URL |
| 405 | `method_not_allowed` | 지원하지 않는 HTTP method |
| 원래 상태 | `http_error` | 기타 Werkzeug HTTP 오류 |
| 500 | `internal_error` | 분류되지 않은 서버 오류 |

내부 예외 메시지와 스택은 응답에 노출하지 않습니다.

## 캐시와 정리

- 시작 시 저장소 스캔은 심볼릭 링크 모델 디렉터리와 `.keras` 파일을 등록하지 않습니다.
- 모델 캐시는 프로세스별 `OrderedDict` 기반 LRU입니다.
- 캐시 히트는 항목을 가장 최근 위치로 이동합니다.
- 캐시 미스는 결정적인 순서로 발견한 첫 `.keras` 파일을 `compile=False`, `safe_mode=True`로 로드합니다.
- 모델은 예측 시작 전에 마지막 사용 시각을 갱신합니다.
- 마지막 사용 시각이 정확히 1주보다 오래된 모델은 정리 대상입니다.
- 정리는 활성 디렉터리와 교체 백업을 삭제한 뒤 메타데이터와 캐시를 제거합니다.

## 런타임 설정

| 환경 변수 | 기본값 | 제약 |
|---|---:|---|
| `MODEL_STORE_PATH` | `data/model_` | 쓰기 가능한 경로 |
| `MAX_MODEL_FILE_SIZE` | `104857600` | 1 이상 정수 |
| `MAX_CACHED_MODELS` | `10` | 1 이상 정수 |
| `MODEL_CLEANUP_INTERVAL` | `5` | 0 이상 시간 단위 정수 |
| `LOG_LEVEL` | `INFO` | Python 로그 레벨 |
| `LOKI_URL` | 미설정 | 선택 사항 |
| `APP_NAME` | `ml-api` | 메트릭/로그 태그 |
| `APP_ENV` | `dev` | 메트릭/로그 태그 |
| `SERVER_HOST` | `0.0.0.0` | 개발 서버 전용 |
| `SERVER_PORT` | `5000` | 1~65535 |

정수 환경 변수가 형식 또는 범위를 위반하면 애플리케이션 시작이 실패합니다.

## 배포 계약

- 공식 컨테이너는 CPU 전용 런타임입니다(Linux x86-64: `tensorflow-cpu`, ARM64: `tensorflow`).
- 컨테이너는 Gunicorn worker 1개로 실행합니다.
- 컨테이너 포트는 `5000`입니다.
- Compose의 기본 호스트 포트는 `12021`입니다.
- Compose는 기존 호스트 `./data`를 컨테이너 `/app/data`에 연결합니다. 기본 모델 디렉터리는 `data/model_`이며 런타임 UID/GID는 10001:10001입니다.
- 여러 worker/replica 간에는 메타데이터, 캐시, 락을 공유하지 않습니다.
