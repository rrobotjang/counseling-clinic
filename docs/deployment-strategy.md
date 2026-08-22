# 심리상담 클리닉 - 프로덕션 배포 전략

## 아키텍처 개요

### Phase 1: 300 동시성 (현재 상태)
```
[사용자] → [FastAPI + WebSocket (port 7862)] → [InferencePool]
                                                    ↓
                                          [ollama qwen2.5:3b (CPU)]
                                                    ↓
                                          [Cloud API 폴백 (Clova)]
```

**캐시 계층:**
- Level 1: TemplatePrecompute (45개 패턴, ~2μs, 즉시)
- Level 2: ResponseCache (로컬 해시, ~0.1ms, TTL 5분)
- Level 3: Redis (멀티 인스턴스 공유, ~1ms, TTL 10분)
- Level 4: InferencePool (ollama CPU ~3.5초, Clova 폴백 ~1초)

**예상 성능:**
- 캐시 히트: ~70% (템플릿 45개 + 응답 캐시)
- ollama 요청: ~30% = 90명/300 동시
- ollama 처리: 90 × 3.5초 = 315초 = ~5분 (전체 순회)
- 체감 대기: 첫 요청 3.5초, 이후 캐시로 즉시

### Phase 2: 1000 동시성
```
[사용자] → [Nginx LB] → [FastAPI ×2] → [InferencePool] → [Redis 캐시]
                ↓              ↓                            ↓
           [TLS 종료]     [ollama GPU]              [데이터 공유]
```

**드래프트 변경:**
- FastAPI ×2 인스턴스 (least_conn 로드밸런싱)
- Redis 캐시 공유 (동일 응답 across 인스턴스)
- ollama GPU (NVIDIA T4/A10G: ~0.5초/요청)
- Nginx: TLS 종료 + Rate Limiting + 정적 파일

---

## 배포 옵션 비교

| 항목 | Render (현재) | AWS/GCP GPU |-Colab Pro |
|------|---------------|-------------|-----------|
| 비용 | 무료~20$/월 | 200~500$/월 | 10$/월 |
| GPU | 없음 | NVIDIA T4/A10G | NVIDIA T4 |
| ollama | CPU (3.5초) | GPU (0.5초) | GPU (0.5초) |
| 동시성 | ~50 | ~500 | ~100 |
| 설정 | 쉬움 | 복잡 | 중간 |

---

## Render 배포 가이드 (현재 목표)

### 1. 환경변수 설정
```bash
# Render 대시보드 > Environment
OLLAMA_MODELS=/data/ollama/models
CLOVA_API_URL=https://clovastudio.apigw.ntruss.com
CLOVA_API_KEY=your-api-key-here
REDIS_URL=redis://default:password@redis-cloud.com:12345
ALLOWED_ORIGINS=https://counseling-clinic.onrender.com
```

### 2. ollama 모델 사전 설치
Render는 ollama가 모델을 다운로드할 시간이 없으므로:
1. Docker 이미지에 모델 포함 (Dockerfile에서 `ollama pull`)
2. 또는 외부 Redis에 모델 캐시 공유

### 3. 무료 티어 제한
- 512MB RAM → qwen2.5:3b (1.9GB) 불가
- 750시간/월 → 24/7 구동 시 약 31일
- 콜드 스타트: 첫 요청 시 30초~1분 지연

### 4. 권장 설정
```yaml
# render.yaml
services:
  - type: web
    name: counseling-clinic
    runtime: python
    plan: starter  # $7/월, 512MB RAM
    buildCommand: pip install -r requirements.txt
    startCommand: python app_render_v2.py --port $PORT
    envVars:
      - key: OLLAMA_MODELS
        sync: false
      - key: ALLOWED_ORIGINS
        value: https://counseling-clinic.onrender.com
```

---

## 1000명 확장 시 필수 항목

### Priority 1: GPU 서버
- ollama GPU: 0.5초/요청 (CPU 대비 7배)
- 추천: AWS G4dn.xlarge (NVIDIA T4, ~0.5$/시간)
- 또는: GCP g2-standard-4 (NVIDIA L4)

### Priority 2: Redis 클러스터
- 현재: Redis Cloud 무료 티어 (30MB)
- 확장: Redis Cluster (100MB+, ~25$/월)
- 역할: 응답 캐시 + 세션 관리 + Rate Limiting

### Priority 3: 모니터링
- Prometheus + Grafana (메트릭)
- Sentry (에러 추적)
- 또는 Render 메트릭 (내장)

### Priority 4: CI/CD
- GitHub Actions → Docker Build → Render 배포
- 롤백: 이전 커밋으로 즉시 복귀

---

## 로드맵

| 단계 | 동시성 | 인프라 | 비용 |
|------|--------|--------|------|
| 현재 | 50 | Render 무료 | 0$/월 |
| Phase 1 | 300 | Render Starter + Redis | 12$/월 |
| Phase 2 | 500 | AWS G4dn + Redis | 200$/월 |
| Phase 3 | 1000 | AWS G4dn ×2 + Redis Cluster | 400$/월 |

---

## 모니터링 대시보드

### 필수 메트릭
1. **응답 시간**: p50, p95, p99
2. **캐시 히트율**: 템플릿, 응답, Redis
3. **ollama 대기열**: 현재 큐 크기, 평균 대기 시간
4. **에러율**: 4xx, 5xx
5. **연결 수**: WebSocket 동시 연결

### 알림 기준
- 응답 시간 p95 > 5초 → GPU 업그레이드 검토
- 캐시 히트율 < 50% → 템플릿 패턴 확대
- ollama 대기열 > 100 → 수평 확장 검토
- 에러율 > 1% → 즉시 조치
