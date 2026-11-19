---
title: Steam KO Review Aspect Analyzer
emoji: 🎮
colorFrom: indigo
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# Steam 한국어 리뷰 속성 분석기

Steam 한국어 리뷰를 8개 속성(그래픽·최적화·스토리·게임성·조작감·사운드·가격·운영)별 미언급/긍정/보통/부정과
종합 평가(추천/비추천)로 분석합니다. 직접 학습한 KLUE-RoBERTa 멀티태스크 모델과 TF-IDF+LR 베이스라인을 사용합니다.

- 모델 파일: Space 변수 `MODEL_REPO`에 지정한 Hugging Face 모델 저장소에서 내려받습니다.
- `preprocess.py`, `absa_model.py`는 프로젝트 `src/`의 복사본입니다(학습과 같은 전처리·모델 코드).
