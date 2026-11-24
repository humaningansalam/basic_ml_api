# Repository Guide

이 파일은 저장소 전체에 적용됩니다.

## 먼저 읽을 문서

- 사용과 운영: `README.md`
- 외부 계약: `docs/SPEC.md`
- 내부 경계와 불변조건: `docs/ARCHITECTURE.md`

## 코드 지도

- `src/main.py`: 조립만 담당
- `src/api/`: HTTP 파싱과 응답 변환
- `src/core/model_archive.py`: ZIP 검증/추출
- `src/core/model_manager.py`: 저장소, 캐시, 교체, 정리, 락
- `src/core/model_types.py`: 결과 타입과 JSON 변환
- `src/common/`: 시계, 오류, 메트릭
- `tests/`: 공개 계약과 복구/동시성 회귀

## 필수 명령

```bash
uv sync --frozen --group dev
uv run ruff format --check .
uv run ruff check .
uv run python -m pytest -q
```

## 가드레일

- URL, 응답 envelope, `error.code`를 조용히 바꾸지 않습니다.
- Flask 요청 객체를 `src/core`로 전달하지 않습니다.
- ZIP 정책을 `ModelManager`에 다시 섞지 않습니다.
- 상태 변경은 기존 전역 락과 해시별 락의 순서를 지킵니다.
- naive datetime과 로컬 타임존 기반 비즈니스 규칙을 추가하지 않습니다.
- 내부 예외 문자열을 HTTP 응답에 노출하지 않습니다.
- 새 의존성, 자동화, 템플릿은 실제 유지보수 비용을 줄이는 근거가 있을 때만 추가합니다.
- 코드 변경에는 가장 가까운 회귀 테스트와 해당 소유 문서 변경을 함께 포함합니다.
