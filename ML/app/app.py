"""Steam 한국어 리뷰 속성 분석기 — Gradio 앱 (Hugging Face Spaces 배포용)

직접 학습한 두 모델을 불러와 추론한다.
  - KLUE-RoBERTa 멀티태스크 (제안 모델): absa_klue/ (config·tokenizer·model.pt·absa_meta.json)
  - TF-IDF + 로지스틱 회귀 (베이스라인): baseline.joblib
모델 파일 위치 (먼저 찾은 곳 사용):
  1) 환경변수 MODEL_DIR  2) app 폴더 옆 models/  3) HF Hub 저장소(환경변수 MODEL_REPO)를 내려받음
"""
import html
import os
import re
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "src"))   # 로컬 개발: 프로젝트 src/ 사용
from preprocess import clean_text, hangul_ratio  # noqa: E402

ASPECTS = ["graphics", "performance", "story", "gameplay", "controls", "sound", "price", "operation"]
ASPECTS_KO = ["그래픽", "최적화", "스토리", "게임성", "조작감", "사운드", "가격", "운영"]
LABEL_KO = ["미언급", "긍정", "보통", "부정"]
COLORS = {1: "#2e9d5b", 2: "#d49a1a", 3: "#d4483b"}
LOW_CONF = 0.60          # 이보다 확신도가 낮으면 '애매함' 표시
MIN_CHARS, MAX_CHARS = 5, 2000
MODEL_REPO = os.environ.get("MODEL_REPO", "your-hf-id/steam-ko-absa-klue")   # ▶ 배포 시 Space 변수로 설정
KLUE_NAME, BASE_NAME = "KLUE-RoBERTa (제안 모델)", "TF-IDF + LR (베이스라인)"


# ───────────────────────── 모델 로딩 ─────────────────────────
def find_model_dir():
    cands = [os.environ.get("MODEL_DIR"), HERE / "models", HERE.parent / "models"]
    for c in cands:
        if c and (Path(c) / "absa_klue").exists() or (c and (Path(c) / "baseline.joblib").exists()):
            return Path(c)
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=MODEL_REPO))
    except Exception as e:  # 저장소가 없거나 네트워크 오류
        print("HF Hub 다운로드 실패:", e)
        return None


class KlueBackend:
    def __init__(self, model_dir: Path):
        import torch
        from absa_model import AbsaModel
        self.torch = torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        path = model_dir / "absa_klue" if (model_dir / "absa_klue").exists() else model_dir
        self.model, self.tok, self.meta = AbsaModel.load(path, device="cpu")

    def predict(self, texts):
        from absa_model import predict_proba
        return predict_proba(self.model, self.tok, texts, max_len=self.meta["max_len"], batch_size=16)


class BaselineBackend:
    def __init__(self, model_dir: Path):
        import joblib
        self.m = joblib.load(model_dir / "baseline.joblib")

    def predict(self, texts):
        X = self.m["vectorizer"].transform(texts)
        pa = np.zeros((len(texts), len(ASPECTS), 4))
        for j, a in enumerate(ASPECTS):
            clf = self.m["clfs"][a]
            pa[:, j, clf.classes_.astype(int)] = clf.predict_proba(X)   # 학습에 없던 클래스는 확률 0
        po = self.m["overall_clf"].predict_proba(self.m["overall_vectorizer"].transform(texts))
        return pa, po


BACKENDS, LOAD_ERRORS = {}, []
_model_dir = find_model_dir()
if _model_dir is not None:
    for name, cls in [(KLUE_NAME, KlueBackend), (BASE_NAME, BaselineBackend)]:
        try:
            t0 = time.time()
            BACKENDS[name] = cls(_model_dir)
            print(f"로드 완료: {name} ({time.time() - t0:.1f}s)")
        except Exception as e:
            LOAD_ERRORS.append(f"{name}: {type(e).__name__}: {e}")
            print("로드 실패:", LOAD_ERRORS[-1])
MODEL_CHOICES = list(BACKENDS) or ["(사용 가능한 모델 없음)"]


def get_backend(name):
    if name not in BACKENDS:
        raise gr.Error("모델을 불러오지 못했습니다. " + " / ".join(LOAD_ERRORS or ["모델 파일 위치를 확인하세요."]))
    return BACKENDS[name]


# ───────────────────────── 화면 구성 요소 ─────────────────────────
CSS_CARD = "border:1px solid rgba(128,128,128,.35);border-radius:12px;padding:16px 18px;margin:6px 0;"


def bar(p, color, width=180):
    return (f'<span style="display:inline-block;width:{width}px;height:10px;border-radius:5px;'
            f'background:rgba(128,128,128,.2);vertical-align:middle;">'
            f'<span style="display:block;width:{p * 100:.0f}%;height:100%;border-radius:5px;background:{color};"></span></span>')


def render_single(pa, po, notes):
    p_rec = float(po[1])
    icon, verdict, col = ("👍", "추천", COLORS[1]) if p_rec >= 0.5 else ("👎", "비추천", COLORS[3])
    out = [f'<div style="{CSS_CARD}"><div style="font-size:13px;opacity:.7;">종합 평가</div>'
           f'<div style="font-size:24px;font-weight:700;color:{col};">{icon} {verdict}'
           f'<span style="font-size:15px;font-weight:400;opacity:.8;"> &nbsp;추천 확률 {p_rec:.0%}</span></div></div>']
    pred = pa.argmax(-1)
    mentioned = [j for j in range(len(ASPECTS_KO)) if pred[j] != 0]
    out.append(f'<div style="{CSS_CARD}"><div style="font-size:13px;opacity:.7;margin-bottom:8px;">언급된 속성</div>')
    if not mentioned:
        out.append('<div style="opacity:.8;">구체적으로 언급된 속성이 없습니다 — 리뷰 전체의 호불호는 위 종합 평가를 참고하세요.</div>')
    for j in sorted(mentioned, key=lambda j: -pa[j].max()):
        c, conf = int(pred[j]), float(pa[j].max())
        amb = ' <span style="font-size:12px;padding:1px 6px;border-radius:8px;background:rgba(128,128,128,.25);">애매함</span>' if conf < LOW_CONF else ""
        detail = " · ".join(f"{LABEL_KO[k]} {pa[j, k]:.0%}" for k in (1, 2, 3))
        out.append(f'<div style="margin:6px 0;"><b style="display:inline-block;width:64px;">{ASPECTS_KO[j]}</b>'
                   f'<b style="display:inline-block;width:44px;color:{COLORS[c]};">{LABEL_KO[c]}</b>'
                   f'{bar(conf, COLORS[c])} <span style="display:inline-block;width:42px;text-align:right;">{conf:.0%}</span>{amb}'
                   f'<div style="font-size:12px;opacity:.6;margin-left:112px;">{detail}</div></div>')
    none = [ASPECTS_KO[j] for j in range(len(ASPECTS_KO)) if pred[j] == 0]
    if none:
        out.append(f'<div style="font-size:13px;opacity:.6;margin-top:10px;">언급되지 않음: {" · ".join(none)}</div>')
    out.append("</div>")
    for n in notes:
        out.append(f'<div style="font-size:13px;opacity:.75;">ⓘ {html.escape(n)}</div>')
    return "".join(out)


def render_aggregate(pa, po, voted, game, n):
    pred = pa.argmax(-1)
    p_rec_model = float((po[:, 1] >= 0.5).mean())
    real = f" · 실제 추천 버튼 비율 <b>{np.mean(voted):.0%}</b>" if len(voted) else ""
    out = [f'<div style="{CSS_CARD}"><div style="font-size:13px;opacity:.7;">{html.escape(game)} · 최근 한국어 리뷰 {n}건</div>'
           f'<div style="font-size:20px;font-weight:700;">모델이 예측한 추천 비율 {p_rec_model:.0%}'
           f'<span style="font-size:14px;font-weight:400;opacity:.8;">{real}</span></div></div>',
           f'<div style="{CSS_CARD}"><div style="font-size:13px;opacity:.7;margin-bottom:8px;">'
           f'속성별 평판 — 언급률 높은 순 (막대: 언급한 리뷰 중 긍정/보통/부정 비율)</div>']
    order = np.argsort(-(pred != 0).mean(0))
    for j in order:
        m = pred[:, j] != 0
        rate = m.mean()
        if m.sum() == 0:
            segs = '<span style="opacity:.5;font-size:13px;">언급 없음</span>'
        else:
            segs = '<span style="display:inline-flex;width:260px;height:14px;border-radius:7px;overflow:hidden;vertical-align:middle;">'
            for k in (1, 2, 3):
                share = (pred[m, j] == k).mean()
                segs += f'<span title="{LABEL_KO[k]} {share:.0%}" style="width:{share * 100:.1f}%;background:{COLORS[k]};"></span>'
            segs += "</span> " + " ".join(f'<span style="font-size:12px;color:{COLORS[k]};">{LABEL_KO[k]} {(pred[m, j] == k).mean():.0%}</span>' for k in (1, 2, 3))
        out.append(f'<div style="margin:7px 0;"><b style="display:inline-block;width:64px;">{ASPECTS_KO[j]}</b>'
                   f'<span style="display:inline-block;width:96px;font-size:13px;opacity:.8;">언급 {rate:.0%} ({int(m.sum())}건)</span>{segs}</div>')
    out.append("</div>")
    return "".join(out)


# ───────────────────────── 기능 1: 리뷰 한 건 분석 ─────────────────────────
def analyze_text(text, model_name):
    if text is None or not text.strip():
        raise gr.Error("리뷰 내용을 입력해 주세요.")
    notes = []
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
        notes.append(f"입력이 길어 앞 {MAX_CHARS}자만 분석했습니다.")
    clean = clean_text(text)
    if len(clean) < MIN_CHARS:
        raise gr.Error(f"정제 후 {len(clean)}자로 너무 짧습니다. 한 문장 이상 입력해 주세요.")
    if hangul_ratio(clean) < 0.3:
        notes.append("한국어 리뷰로 학습한 모델이라 한국어가 아닌 입력은 정확도가 낮을 수 있습니다.")
    pa, po = get_backend(model_name).predict([clean])
    notes.append(f"모델: {model_name}")
    return render_single(pa[0], po[0], notes)


# ───────────────────────── 기능 2: 게임(App ID) 분석 ─────────────────────────
STEAM_REVIEWS = "https://store.steampowered.com/ajaxappreviews/{appid}"   # 01_collect와 같은 공개 엔드포인트
HEADERS = {"User-Agent": "Mozilla/5.0 (academic-project review analyzer)"}


def parse_appid(q):
    q = (q or "").strip()
    m = re.search(r"/app/(\d+)", q) or re.fullmatch(r"(\d{1,8})", q)
    if not m:
        raise gr.Error("Steam App ID(숫자) 또는 상점 페이지 주소를 입력해 주세요. 예: 1086940 또는 https://store.steampowered.com/app/1086940/")
    return int(m.group(1))


def game_name(appid):
    try:
        r = requests.get("https://store.steampowered.com/api/appdetails",
                         params={"appids": appid, "l": "koreana"}, headers=HEADERS, timeout=10)
        d = r.json().get(str(appid), {})
        return d["data"]["name"] if d.get("success") else f"App {appid}"
    except Exception:
        return f"App {appid}"


def fetch_reviews(appid, n, progress):
    out, cursor, seen = [], "*", set()
    while len(out) < n:
        params = {"json": 1, "language": "koreana", "filter": "recent", "review_type": "all",
                  "purchase_type": "all", "num_per_page": 100, "cursor": cursor}
        try:
            r = requests.get(STEAM_REVIEWS.format(appid=appid), params=params, headers=HEADERS, timeout=15)
            if r.status_code == 429:
                raise gr.Error("Steam 요청이 많아 잠시 막혔습니다. 1분 뒤 다시 시도해 주세요.")
            data = r.json()
        except gr.Error:
            raise
        except Exception:
            raise gr.Error("Steam 리뷰를 가져오지 못했습니다. App ID를 확인하거나 잠시 뒤 다시 시도해 주세요.")
        if data.get("success") != 1:
            raise gr.Error("존재하지 않는 App ID이거나 리뷰를 불러올 수 없는 게임입니다.")
        reviews = data.get("reviews", [])
        new = [rv for rv in reviews if rv["recommendationid"] not in seen]
        if not new:
            break
        for rv in new:
            seen.add(rv["recommendationid"])
            out.append(rv)
        cursor = data.get("cursor", cursor)
        progress(min(len(out), n) / n * 0.5, desc=f"리뷰 수집 {min(len(out), n)}/{n}")
        time.sleep(0.5)
    return out[:n]


def analyze_game(query, n, model_name, progress=gr.Progress()):
    appid = parse_appid(query)
    backend = get_backend(model_name)
    progress(0, desc="게임 정보 확인")
    name = game_name(appid)
    raw = fetch_reviews(appid, int(n), progress)
    rows = [(clean_text(rv.get("review", "")), bool(rv.get("voted_up"))) for rv in raw]
    rows = [(t, v) for t, v in rows if len(t) >= MIN_CHARS]
    if len(rows) < 5:
        raise gr.Error(f"{name}: 분석할 수 있는 한국어 리뷰가 {len(rows)}건뿐입니다. 한국어 리뷰가 많은 게임으로 시도해 주세요.")
    texts, voted = [t for t, _ in rows], [v for _, v in rows]
    progress(0.6, desc=f"{len(texts)}건 분석 중")
    pa, po = backend.predict(texts)
    pred = pa.argmax(-1)
    table = pd.DataFrame({
        "리뷰": [t[:80] + ("…" if len(t) > 80 else "") for t in texts],
        "종합(예측)": ["추천" if p >= 0.5 else "비추천" for p in po[:, 1]],
        "실제 버튼": ["추천" if v else "비추천" for v in voted],
        "언급 속성": [", ".join(f"{ASPECTS_KO[j]}:{LABEL_KO[pred[i, j]]}" for j in range(len(ASPECTS)) if pred[i, j]) or "-"
                    for i in range(len(texts))],
    })
    return render_aggregate(pa, po, voted, name, len(texts)), table


# ───────────────────────── UI ─────────────────────────
EXAMPLES = ["그래픽은 정말 예쁜데 최적화가 엉망이라 프레임이 계속 떨어져요",
            "스토리랑 OST가 미쳤다. 가격도 할인하면 혜자",
            "서버 터지고 핵쟁이 천지. 운영 좀 제대로 해라",
            "소매치기만 하고 다니다 보니, 현실에서도 하고 싶어요.."]

GUIDE = f"""
**무엇을 하나요?** Steam 한국어 리뷰를 읽고 **8개 속성**(그래픽·최적화·스토리·게임성·조작감·사운드·가격·운영)마다
미언급 / 긍정 / 보통 / 부정을 판정하고, 리뷰 전체의 **추천 여부**(종합 평가)를 예측합니다.

- **리뷰 분석:** 리뷰 문장을 붙여 넣고 `분석하기`. 막대는 모델의 확신도이며 {LOW_CONF:.0%} 미만이면 *애매함*으로 표시합니다.
- **게임 분석:** Steam App ID(예: `1086940` 발더스 게이트 3) 또는 상점 주소를 넣으면 최근 한국어 리뷰를 모아 속성별 평판을 보여줍니다.
- 두 모델 모두 직접 학습했습니다 — **KLUE-RoBERTa 멀티태스크**(제안 모델), **TF-IDF + 로지스틱 회귀**(베이스라인).
- 한계: LLM으로 만든 약라벨로 학습했고, 사운드처럼 언급이 드문 속성은 정확도가 낮습니다. 비꼬는 말투·은유는 틀릴 수 있습니다.
"""

with gr.Blocks(title="Steam 한국어 리뷰 속성 분석기", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎮 Steam 한국어 리뷰 속성 분석기\nSteam은 추천 비율 하나만 보여줍니다. 이 앱은 리뷰를 **속성별 평가**로 나눠 보여줍니다.")
    with gr.Accordion("사용 안내", open=False):
        gr.Markdown(GUIDE)
    if LOAD_ERRORS:
        gr.Markdown("⚠ 일부 모델을 불러오지 못했습니다: " + " / ".join(LOAD_ERRORS))
    model_sel = gr.Radio(MODEL_CHOICES, value=MODEL_CHOICES[0], label="모델")

    with gr.Tab("리뷰 분석"):
        with gr.Row():
            with gr.Column(scale=1):
                txt = gr.Textbox(label="리뷰 입력", lines=6, max_lines=12,
                                 placeholder="예) 그래픽은 정말 예쁜데 최적화가 엉망이라 프레임이 계속 떨어져요")
                btn = gr.Button("분석하기", variant="primary")
                gr.Examples(EXAMPLES, inputs=txt)
            with gr.Column(scale=1):
                out_single = gr.HTML()
        btn.click(analyze_text, [txt, model_sel], out_single)
        txt.submit(analyze_text, [txt, model_sel], out_single)

    with gr.Tab("게임 분석 (App ID)"):
        with gr.Row():
            q = gr.Textbox(label="Steam App ID 또는 상점 주소", placeholder="1086940", scale=3)
            n = gr.Slider(30, 200, value=100, step=10, label="분석할 최근 리뷰 수", scale=2)
        btn2 = gr.Button("게임 분석하기", variant="primary")
        out_agg = gr.HTML()
        out_tbl = gr.Dataframe(label="리뷰별 예측", wrap=True)
        gr.Examples([["1086940"], ["1245620"], ["2246340"], ["413150"]], inputs=q)
        btn2.click(analyze_game, [q, n, model_sel], [out_agg, out_tbl])

if __name__ == "__main__":
    demo.launch()
