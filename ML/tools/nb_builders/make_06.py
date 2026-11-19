import sys
from nbtools import NB, SETUP

nb = NB()
nb.md("""
# 06. 배포 — Hugging Face 모델 업로드 + Spaces 앱 배포

**런타임: CPU로 충분** (학습 없음, 05에서 저장한 가중치를 올리기만 한다). 약 5분.

| 단계 | 내용 |
|---|---|
| ① 사전 확인 | 저장된 가중치(`models/absa_klue`, `models/baseline.joblib`)를 **실제로 불러와 추론**해 본다 — 깨진 파일을 올리지 않기 위해 |
| ② 모델 업로드 | HF 모델 저장소 `HF아이디/steam-ko-absa-klue` (공개) |
| ③ 앱 배포 | HF Space `HF아이디/steam-ko-review-analyzer` 생성 → `app/` 업로드 → Space 변수 `MODEL_REPO` 설정 |

### 처음 한 번만: Hugging Face 준비

1. https://huggingface.co/join 에서 가입 (이메일 인증까지)
2. 오른쪽 위 프로필 → **Settings → Access Tokens → + Create new token**
   - Token type: **Write** → 이름 아무거나 → Create → 토큰 복사 (`hf_...`)
3. Colab 왼쪽 세로 메뉴의 **🔑 (보안 비밀/Secrets)** → **+ 새 보안 비밀 추가**
   - 이름 `HF_TOKEN`, 값에 복사한 토큰 붙여넣기, **노트북 액세스 토글 ON**
4. 아래 설정 셀의 `HF_ID`를 본인 HF 아이디(프로필 주소 `huggingface.co/아이디`)로 수정

> 토큰은 코드나 GitHub에 절대 쓰지 않는다 (평가표: API Key 업로드 금지).
""")

nb.code(SETUP)

nb.code("""
HF_ID = "your-hf-id"                                  # ▶ 본인 HF 아이디로 수정
MODEL_REPO = f"{HF_ID}/steam-ko-absa-klue"            # 모델 저장소
SPACE_REPO = f"{HF_ID}/steam-ko-review-analyzer"      # 앱(Space) → 데모 URL

MODEL_DIR = ROOT / "models"
APP_DIR = ROOT / "app"
assert HF_ID != "your-hf-id", "HF_ID를 본인 아이디로 바꿔 주세요."
for p in [MODEL_DIR / "absa_klue" / "model.pt", MODEL_DIR / "absa_klue" / "absa_meta.json",
          MODEL_DIR / "baseline.joblib", APP_DIR / "app.py", APP_DIR / "requirements.txt", APP_DIR / "README.md"]:
    assert p.exists(), f"파일이 없습니다: {p}"
    print(f"✔ {p.relative_to(ROOT)}  ({p.stat().st_size / 1e6:.1f} MB)")
""")

nb.md("## ① 사전 확인 — 저장된 가중치로 실제 추론")

nb.code("""
import shutil, joblib
from absa_model import AbsaModel, predict_proba
from preprocess import clean_text

LABEL = ["미언급", "긍정", "보통", "부정"]
ASPECTS_KO = ["그래픽", "최적화", "스토리", "게임성", "조작감", "사운드", "가격", "운영"]
s = "그래픽은 정말 예쁜데 최적화가 엉망이라 프레임이 계속 떨어져요"

m, tok, meta = AbsaModel.load(MODEL_DIR / "absa_klue", device="cpu")
pa, po = predict_proba(m, tok, [clean_text(s)], max_len=meta["max_len"])
print("KLUE  :", {k: LABEL[v] for k, v in zip(ASPECTS_KO, pa[0].argmax(-1)) if v}, f"| 추천 {po[0, 1]:.0%}")

b = joblib.load(MODEL_DIR / "baseline.joblib")
x = b["vectorizer"].transform([clean_text(s)])
print("베이스라인:", {k: LABEL[int(b["clfs"][a].predict(x)[0])] for a, k in zip(b["aspects"], ASPECTS_KO)
                  if int(b["clfs"][a].predict(x)[0])})
print("테스트 지표(저장 시점):", meta.get("test_metrics"))
""")

nb.md("## ② 모델 업로드 · ③ 앱 배포")

nb.code("""
from huggingface_hub import HfApi, login
from google.colab import userdata

login(token=userdata.get("HF_TOKEN"))
api = HfApi()
print("로그인:", api.whoami()["name"])

# ② 모델 — 공개 저장소(Space가 토큰 없이 내려받음). absa_klue 내용은 저장소 루트에, baseline.joblib도 루트에
api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=str(MODEL_DIR / "absa_klue"), repo_id=MODEL_REPO, repo_type="model",
                  commit_message="KLUE-RoBERTa multitask ABSA weights")
api.upload_file(path_or_fileobj=str(MODEL_DIR / "baseline.joblib"), path_in_repo="baseline.joblib",
                repo_id=MODEL_REPO, repo_type="model", commit_message="TF-IDF + LR baseline")
print("모델 업로드 완료:", f"https://huggingface.co/{MODEL_REPO}")

# ③ 앱 — src의 최신 전처리·모델 코드를 app/에 복사한 뒤 업로드
for f in ["preprocess.py", "absa_model.py"]:
    shutil.copy(ROOT / "src" / f, APP_DIR / f)
api.create_repo(SPACE_REPO, repo_type="space", space_sdk="gradio", exist_ok=True)
api.add_space_variable(SPACE_REPO, "MODEL_REPO", MODEL_REPO)
api.upload_folder(folder_path=str(APP_DIR), repo_id=SPACE_REPO, repo_type="space",
                  ignore_patterns=["__pycache__/*", "*.pyc", "models/*", ".gitkeep"],
                  commit_message="Deploy Gradio app")
print("앱 배포 완료")
print("\\n데모 URL:", f"https://huggingface.co/spaces/{SPACE_REPO}")
print("→ 빌드에 5~10분 걸립니다. Space 페이지 상단이 'Running'이 되면 접속하세요.")
""")

nb.code("""
# Space 빌드 상태 확인 (몇 분 뒤 다시 실행) — RUNNING이면 완료, *_ERROR면 Space 페이지의 Logs 탭 확인
rt = api.get_space_runtime(SPACE_REPO)
print("상태:", rt.stage, "| 하드웨어:", rt.hardware)
""")

nb.md("""
## 배포 후 할 일

1. 데모 URL 접속 → **화면 캡처 3장**을 `docs/screenshots/`에 저장
   - `01_review.png`: 리뷰 분석 결과 (예시 문장 클릭)
   - `02_edge.png`: *애매함* 표시가 있는 결과 또는 빈 입력 오류
   - `03_game.png`: 게임 분석 탭 (`1086940` 등)
2. README `## 0`, `## 7`, `## 8`과 `submission.json`의 【】 칸에 데모 URL·모델 저장소 주소 기입

**문제가 생기면** Space 페이지 → **Logs** 탭(Build / Container) 내용을 확인한다.
- `MODEL_REPO`를 못 찾음 → ③의 `add_space_variable` 다시 실행 후 Space **Restart**
- 메모리 부족 → 무료 CPU(16GB)로 충분하지만, 게임 분석 리뷰 수를 줄여서 시도
""")

nb.save(sys.argv[1])
