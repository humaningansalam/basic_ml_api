# Contributing

## 개발 환경

Python 3.11과 uv를 사용합니다. 저장소 루트에서 다음 명령으로 고정된 의존성을 설치합니다.

```bash
uv sync --frozen --group dev
```

## 변경 원칙

- 한 변경에는 한 가지 명확한 이유가 있어야 합니다.
- 공개 API envelope와 `error.code`는 호환성 계약입니다.
- HTTP 관심사는 `src/api`, 아카이브 정책은 `src/core/model_archive.py`, 모델 상태 전이는 `src/core/model_manager.py`에 둡니다.
- 이름은 역할과 단위를 드러냅니다. `data`, `obj`, `tmp`, `manager2` 같은 맥락 없는 이름을 새로 만들지 않습니다.
- 시간은 timezone-aware UTC를 사용합니다.
- 실패를 삼키지 않습니다. 클라이언트 오류는 `ApplicationError`, 예상하지 못한 오류는 로그와 안정적인 500 응답으로 남깁니다.
- 새 의존성을 추가할 때는 표준 라이브러리나 기존 경계로 해결할 수 없는 이유를 PR에 적고 `uv.lock`을 함께 갱신합니다.
- 반복 설명을 여러 문서에 복제하지 않습니다. 사용법은 README, 외부 계약은 SPEC, 내부 구조는 ARCHITECTURE가 소유합니다.
- 실제 팀 흐름이 요구하지 않는 Task/Issue 템플릿이나 생성형 문서 묶음을 추가하지 않습니다.

## 검증

모든 변경은 최소 다음을 통과해야 합니다.

```bash
uv run ruff format --check .
uv run ruff check .
uv run python -m pytest -q
```

Docker 또는 배포 파일을 바꾸면 추가로 실행합니다.

```bash
docker compose config
docker build -t basic-ml-api:local .
```

## 테스트 작성

- 공개 동작을 우선 검증하고 내부 구현에 대한 단언은 동시성·복구 불변조건에 필요한 경우로 제한합니다.
- 정상 경로만 추가하지 말고 잘못된 입력, 저장소 실패, 중단 후 복구를 함께 다룹니다.
- 시간, 파일 크기, 경계값은 테스트에서 명시적으로 고정합니다.
- 전역 메트릭과 백그라운드 스레드가 테스트 간 상태를 누출하지 않게 합니다.

## 완료 조건

PR을 열기 전에 다음을 확인합니다.

- 변경 이유와 범위가 설명되어 있음
- 이름과 책임 경계가 현재 구조에 맞음
- 새 동작과 회귀 위험을 테스트함
- API 또는 운영 동작이 바뀌면 관련 문서를 갱신함
- `uv.lock`과 컨테이너 설정이 재현 가능함
- 임시 파일, 생성 로그, 사용하지 않는 주석·템플릿이 없음
