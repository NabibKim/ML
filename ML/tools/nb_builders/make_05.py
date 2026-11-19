import sys
from nbtools import NB, SETUP

nb = NB()
nb.md("""
# 05. 제안 모델 — KLUE-RoBERTa 멀티태스크 파인튜닝

**런타임: Colab GPU(T4)** — 메뉴 `런타임 → 런타임 유형 변경 → T4 GPU`. 전체 실행 약 10~15분.

| 항목 | 내용 |
|---|---|
| 사전학습 모델 | `klue/roberta-base` — 한국어 위키·뉴스·리뷰 등으로 사전학습된 RoBERTa (KLUE 벤치마크 공식 모델) |
| 구조 | 공유 인코더의 `[CLS]` 벡터 → **속성 헤드 8개**(각 4클래스) + **종합 평가 헤드 1개**(추천/비추천) |
| 학습 방식 | **파인튜닝(Fine-tuning)** — 인코더 전체 + 헤드를 함께 학습 |
| 손실 | 속성별 가중 교차엔트로피 평균 + λ × 종합 평가 교차엔트로피 (λ=0.5) |
| 클래스 불균형 | 속성마다 √(역빈도) 클래스 가중치 — "미언급" 쏠림 완화 |
| 종합 평가 데이터 | `voted_up`(작성자의 추천 버튼)을 정답으로 하는 정제 리뷰 10,677건 중 매 epoch 3,000건 무작위 추출 |
| 모델 선택 | 매 epoch Validation **속성 평균 Macro-F1**이 가장 높은 시점 저장, 3 epoch 개선 없으면 조기 종료 |

**왜 이 모델인가**
- 베이스라인(글자 n-gram)은 단어 표면만 보므로 "그래픽은 예쁜데 최적화가…"처럼 **한 문장 안에서 속성마다 감성이 갈리는** 문맥과, 반어·은어를 구분하지 못한다. 문맥을 양방향으로 읽는 Transformer 인코더가 이 한계를 보완한다.
- 한국어 전용 사전학습 모델이라 다국어 모델보다 토큰화 효율과 성능이 좋고, base 크기라 T4에서 수 분 내 학습된다.

**비교 실험(ablation):** 같은 설정에서 종합 평가 과제를 뺀 **속성 단독 모델**도 학습해, 멀티태스크가 속성 분류에 도움이 되는지 확인한다.

**산출물:** `models/absa_klue/`, `results/klue_*.{json,csv}`, `figures/05_*.png`
""")

nb.code(SETUP)

nb.code("""
import json, time, random, copy
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import confusion_matrix, f1_score

from absa import (ASPECTS, ASPECTS_KO, LABELS, LABEL_NAMES, LABEL_NAMES_EN, OVERALL_NAMES,
                  load_splits, load_overall_splits, evaluate, summary, overall_metrics,
                  human_subset_report)
from absa_model import AbsaModel, predict_proba
from preprocess import clean_text

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEVICE,
      "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "⚠ GPU 없음 — 런타임 유형을 T4로 바꾸세요")

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()

CFG = dict(model_name="klue/roberta-base", max_len=256, batch_size=32, epochs=8, patience=3,
           lr_encoder=3e-5, lr_head=1e-4, weight_decay=0.01, warmup_ratio=0.1,
           overall_lambda=0.5, overall_per_epoch=3000, dropout=0.1)
MODEL_DIR = ROOT / "models"; RES_DIR = ROOT / "results"
MODEL_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)
CFG
""")

nb.md("""
## 1. 데이터 — 04와 동일한 분할

- 속성: Train 1,500 / Val 200 / Test 300 (`src/absa.load_splits`, 테스트 라벨 = LLM 라벨)
- 종합 평가: Test·Val은 속성과 **같은 리뷰**, Train은 나머지 정제 리뷰 전체
""")

nb.code("""
train, val, test, TEST_SOURCE = load_splits(ROOT)
o_tr, o_va, o_te = load_overall_splits(ROOT)

def overall_of(df):
    return df["voted_up"].astype(str).str.lower().eq("true").astype(int).values

# 속성 라벨이 있는 행 (종합 라벨도 함께 가짐)
asp_train = pd.DataFrame({"text": train["text"], "overall": overall_of(train)})
asp_train[ASPECTS] = train[ASPECTS].values
# 종합 라벨만 있는 행 (속성 Train 리뷰는 제외 — 중복 방지), 속성 라벨은 -100(무시)
ovr_only = o_tr[~o_tr["recommendationid"].isin(train["recommendationid"])][["text", "overall"]].reset_index(drop=True)
for a in ASPECTS:
    ovr_only[a] = -100

Y_va, Y_te = val[ASPECTS].values.astype(int), test[ASPECTS].values.astype(int)
O_va, O_te = overall_of(val), overall_of(test)
assert (O_te == o_te["overall"].values).all()
print(f"속성 Train {len(asp_train)} | 종합 전용 풀 {len(ovr_only):,} | Val {len(val)} | Test {len(test)}")
""")

nb.code("""
# 클래스 가중치: 속성별 √(역빈도), 평균 1로 정규화
counts = np.stack([np.bincount(asp_train[a], minlength=4) for a in ASPECTS]) + 1      # (8, 4)
w = np.sqrt(counts.sum(1, keepdims=True) / (4 * counts))
ASPECT_W = torch.tensor(w / w.mean(1, keepdims=True), dtype=torch.float, device=DEVICE)
oc = np.bincount(o_tr["overall"], minlength=2) + 1
OVERALL_W = torch.tensor(np.sqrt(oc.sum() / (2 * oc)), dtype=torch.float, device=DEVICE)
pd.DataFrame(ASPECT_W.cpu().numpy(), index=ASPECTS_KO, columns=[LABEL_NAMES[c] for c in LABELS]).round(2)
""")

nb.md("## 2. 학습 함수")

nb.code("""
tokenizer = AutoTokenizer.from_pretrained(CFG["model_name"])

def batches(df, bs, shuffle, rng=None):
    idx = np.arange(len(df))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        b = df.iloc[idx[i:i + bs]]
        enc = tokenizer(b["text"].tolist(), padding=True, truncation=True,
                        max_length=CFG["max_len"], return_tensors="pt")
        yield (enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE),
               torch.tensor(b[ASPECTS].values, dtype=torch.long, device=DEVICE),
               torch.tensor(b["overall"].values, dtype=torch.long, device=DEVICE))

def compute_loss(a_logits, o_logits, y_asp, y_ovr, lam):
    losses = []
    for j in range(len(ASPECTS)):
        m = y_asp[:, j] != -100
        if m.any():
            losses.append(F.cross_entropy(a_logits[m, j].float(), y_asp[m, j], weight=ASPECT_W[j]))
    l_asp = torch.stack(losses).mean() if losses else a_logits.sum() * 0
    l_ovr = F.cross_entropy(o_logits.float(), y_ovr, weight=OVERALL_W) if lam > 0 else o_logits.sum() * 0
    return l_asp + lam * l_ovr, l_asp.detach(), l_ovr.detach()

def predict_df(model, df):
    pa, po = predict_proba(model, tokenizer, df["text"].tolist(), max_len=CFG["max_len"],
                           batch_size=64, device=DEVICE)
    return pa.argmax(-1), po.argmax(-1), pa, po

def train_model(multitask: bool, tag: str):
    set_seed()
    lam = CFG["overall_lambda"] if multitask else 0.0
    model = AbsaModel.from_pretrained_encoder(CFG["model_name"], dropout=CFG["dropout"]).to(DEVICE)
    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": CFG["lr_encoder"]},
                             {"params": head_params, "lr": CFG["lr_head"]}], weight_decay=CFG["weight_decay"])
    n_rows = len(asp_train) + (CFG["overall_per_epoch"] if multitask else 0)
    steps = CFG["epochs"] * int(np.ceil(n_rows / CFG["batch_size"]))
    sched = get_linear_schedule_with_warmup(opt, int(CFG["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    rng = np.random.RandomState(SEED)

    history, best_f1, best_state, bad = [], -1, None, 0
    for ep in range(1, CFG["epochs"] + 1):
        t0 = time.time(); model.train()
        epoch_df = asp_train
        if multitask:
            epoch_df = pd.concat([asp_train, ovr_only.sample(CFG["overall_per_epoch"], random_state=SEED + ep)],
                                 ignore_index=True)
        tot = [0.0, 0.0, 0.0]; nb_ = 0
        for ids, att, y_a, y_o in batches(epoch_df, CFG["batch_size"], True, rng):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                a_logits, o_logits = model(ids, att)
            loss, la, lo = compute_loss(a_logits, o_logits, y_a, y_o, lam)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot[0] += loss.item(); tot[1] += la.item(); tot[2] += lo.item(); nb_ += 1

        pa, po, *_ = predict_df(model, val)
        vres = summary(evaluate(Y_va, pa))
        vo = overall_metrics(O_va, po)
        row = {"model": tag, "epoch": ep, "train_loss": tot[0] / nb_, "train_aspect_loss": tot[1] / nb_,
               "train_overall_loss": tot[2] / nb_, "val_macro_f1": vres["macro_f1"],
               "val_mention_f1": vres["mention_f1"], "val_overall_macro_f1": vo["overall_macro_f1"],
               "sec": time.time() - t0}
        history.append(row)
        improved = vres["macro_f1"] > best_f1
        if improved:
            best_f1, bad = vres["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"[{tag}] ep{ep} loss {row['train_loss']:.4f} (asp {row['train_aspect_loss']:.4f}, ovr {row['train_overall_loss']:.4f}) "
              f"| val Macro-F1 {row['val_macro_f1']:.4f} 언급F1 {row['val_mention_f1']:.4f} 종합F1 {row['val_overall_macro_f1']:.4f} "
              f"| {row['sec']:.0f}s {'★' if improved else ''}")
        if bad >= CFG["patience"]:
            print(f"[{tag}] 조기 종료 (epoch {ep})")
            break

    model.load_state_dict(best_state)
    print(f"[{tag}] best val Macro-F1 = {best_f1:.4f}")
    return model, pd.DataFrame(history)
""")

nb.md("""
## 3. 학습 (학습 로그)

- ① **제안 모델:** 멀티태스크 (속성 + 종합 평가)
- ② **비교 실험:** 속성 단독 (λ=0, 종합 전용 데이터 미사용)

`RUN_ABLATION=False`로 두면 ②를 건너뛴다(시간 절약).
""")

nb.code("""
RUN_ABLATION = True

model_mt, hist_mt = train_model(multitask=True, tag="KLUE-MT")
pred_mt = predict_df(model_mt, test)
model_mt.cpu()          # GPU 메모리 확보 (저장 시 다시 사용)
torch.cuda.empty_cache()

hist = [hist_mt]
if RUN_ABLATION:
    model_so, hist_so = train_model(multitask=False, tag="KLUE-aspect-only")
    pred_so = predict_df(model_so, test)
    del model_so; torch.cuda.empty_cache()
    hist.append(hist_so)

train_log = pd.concat(hist, ignore_index=True)
train_log.to_csv(RES_DIR / "klue_train_log.csv", index=False)
train_log.round(4)
""")

nb.code("""
# 학습곡선
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for tag, d in train_log.groupby("model", sort=False):
    axes[0].plot(d["epoch"], d["train_aspect_loss"], "o-", label=tag)
    axes[1].plot(d["epoch"], d["val_macro_f1"], "o-", label=tag)
    axes[2].plot(d["epoch"], d["val_mention_f1"], "o-", label=tag)
for ax, t in zip(axes, ["Train aspect loss", "Val aspect Macro-F1", "Val mention-detection F1"]):
    ax.set_title(t); ax.set_xlabel("epoch"); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "05_learning_curve.png", dpi=150, bbox_inches="tight")
plt.show()
""")

nb.md("""
## 4. 테스트셋 평가 및 모델 비교

베이스라인 점수는 04가 저장한 `results/baseline_metrics.json`에서 불러온다. 두 노트북 모두 같은 LLM 테스트 라벨로 채점한다.
""")

nb.code("""
base = json.loads((RES_DIR / "baseline_metrics.json").read_text())
if base["test_label_source"] != TEST_SOURCE:
    print(f"⚠ 테스트 라벨 출처가 다릅니다 (04: {base['test_label_source']}, 05: {TEST_SOURCE}) → 04를 다시 실행하세요.")

res_mt = evaluate(Y_te, pred_mt[0])
rows = [{"모델": "Baseline: TF-IDF + LR", "학습 방식": "Scratch",
         "Macro-F1": base["macro_f1"], "Accuracy": base["accuracy"], "언급 감지 F1": base["mention_f1"],
         "종합 Macro-F1": base["overall_macro_f1"], "종합 Accuracy": base["overall_accuracy"]}]
if RUN_ABLATION:
    res_so = evaluate(Y_te, pred_so[0])
    s = summary(res_so)
    rows.append({"모델": "KLUE-RoBERTa (속성 단독)", "학습 방식": "Fine-tuning",
                 "Macro-F1": s["macro_f1"], "Accuracy": s["accuracy"], "언급 감지 F1": s["mention_f1"],
                 "종합 Macro-F1": np.nan, "종합 Accuracy": np.nan})
s, o = summary(res_mt), overall_metrics(O_te, pred_mt[1])
rows.append({"모델": "KLUE-RoBERTa 멀티태스크 (제안)", "학습 방식": "Fine-tuning",
             "Macro-F1": s["macro_f1"], "Accuracy": s["accuracy"], "언급 감지 F1": s["mention_f1"],
             "종합 Macro-F1": o["overall_macro_f1"], "종합 Accuracy": o["overall_accuracy"]})
compare = pd.DataFrame(rows).set_index("모델")
compare.to_csv(RES_DIR / "model_comparison.csv", encoding="utf-8-sig")
print(f"테스트 라벨 출처: {'사람 검수' if TEST_SOURCE == 'human' else 'LLM'} | 주지표: 속성 평균 Macro-F1")
compare.round(4)
""")

nb.code("""
# [참고] 사람 검수 시도 라벨로 채점했을 때 (최종 평가는 위의 LLM 라벨 기준)
bp_ = pd.read_csv(RES_DIR / "baseline_test_pred.csv", dtype={"recommendationid": str})
preds_h = {"Baseline: TF-IDF + LR": bp_[[f"{a}_pred" for a in ASPECTS]].values}
if RUN_ABLATION:
    preds_h["KLUE-RoBERTa (속성 단독)"] = pred_so[0]
preds_h["KLUE-RoBERTa 멀티태스크 (제안)"] = pred_mt[0]
hs_table, hs_agree, N_HUMAN = human_subset_report(ROOT, test, preds_h)
if N_HUMAN:
    hs_table.to_csv(RES_DIR / "model_comparison_human_subset.csv", encoding="utf-8-sig")
    display(hs_table.round(4))
""")

nb.code("""
# 속성별 Macro-F1 비교
per = pd.DataFrame({"Baseline": pd.Series({a: base["per_aspect"][a]["macro_f1"] for a in ASPECTS})})
if RUN_ABLATION:
    per["KLUE aspect-only"] = res_so.loc[ASPECTS, "macro_f1"].astype(float)
per["KLUE multitask"] = res_mt.loc[ASPECTS, "macro_f1"].astype(float)

ax = per.plot.bar(figsize=(14, 4.5), width=0.8)
ax.set_ylim(0, 1); ax.set_ylabel("test Macro-F1"); ax.grid(axis="y", alpha=0.3)
ax.set_title("Per-aspect Macro-F1 (test)"); ax.set_xticklabels(ASPECTS, rotation=0)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_per_aspect_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
res_mt.round(3)
""")

nb.code("""
# 제안 모델 혼동행렬 (행: 정답, 열: 예측) + 종합 평가
P_te = pred_mt[0]
fig, axes = plt.subplots(3, 3, figsize=(15, 14))
ticks = [LABEL_NAMES_EN[c] for c in LABELS]
for j, (ax, a) in enumerate(zip(axes.ravel(), ASPECTS)):
    cm = confusion_matrix(Y_te[:, j], P_te[:, j], labels=LABELS)
    ax.imshow(cm, cmap="Blues")
    for r in range(4):
        for c in range(4):
            ax.text(c, r, cm[r, c], ha="center", va="center", color="white" if cm[r, c] > cm.max() / 2 else "black")
    ax.set_xticks(range(4), ticks); ax.set_yticks(range(4), ticks)
    ax.set_title(f"{a} (F1={res_mt.loc[a, 'macro_f1']:.2f})"); ax.set_xlabel("pred"); ax.set_ylabel("true")
ax = axes.ravel()[-1]
cm = confusion_matrix(O_te, pred_mt[1], labels=[0, 1])
ax.imshow(cm, cmap="Greens")
for r in range(2):
    for c in range(2):
        ax.text(c, r, cm[r, c], ha="center", va="center", color="white" if cm[r, c] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["not rec.", "rec."]); ax.set_yticks([0, 1], ["not rec.", "rec."])
ax.set_title(f"overall (F1={o['overall_macro_f1']:.2f})"); ax.set_xlabel("pred"); ax.set_ylabel("true")
fig.suptitle("KLUE-RoBERTa multitask — confusion matrices (test)")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
""")

nb.md("""
## 5. 오류 분석

- (a) 제안 모델이 **틀린** 사례 — 어떤 유형이 어려운가
- (b) 베이스라인은 틀리고 제안 모델은 **맞힌** 사례 — 문맥 이해의 효과
""")

nb.code("""
bp = pd.read_csv(RES_DIR / "baseline_test_pred.csv", dtype={"recommendationid": str})
assert (bp["recommendationid"].values == test["recommendationid"].values).all()
B_te = bp[[f"{a}_pred" for a in ASPECTS]].values

def collect(mask):
    out = []
    for i, j in zip(*np.where(mask)):
        out.append({"idx": i, "속성": ASPECTS_KO[j], "정답": LABEL_NAMES[Y_te[i, j]],
                    "Baseline": LABEL_NAMES[B_te[i, j]], "KLUE": LABEL_NAMES[P_te[i, j]],
                    "text": test.loc[i, "text"][:150]})
    return pd.DataFrame(out)

pd.set_option("display.max_colwidth", 150)
wrong = collect(P_te != Y_te)
fixed = collect((B_te != Y_te) & (P_te == Y_te))
print(f"(a) KLUE 오답 {len(wrong)}칸 | (b) Baseline 오답 → KLUE 정답 {len(fixed)}칸 | 반대(KLUE만 오답) {((B_te == Y_te) & (P_te != Y_te)).sum()}칸")
print("\\n(a) 오답 방향 (행: 정답, 열: KLUE 예측)")
print(pd.crosstab(wrong["정답"], wrong["KLUE"]))
wrong.sample(min(10, len(wrong)), random_state=SEED)
""")

nb.code("""
fixed.sample(min(10, len(fixed)), random_state=SEED)
""")

nb.md("""
## 6. 최종 모델 선택 및 저장

테스트 **속성 평균 Macro-F1**(주지표)이 가장 높은 모델을 최종 모델로 한다. 종합 평가 헤드를 가진 멀티태스크 모델이 서비스에 쓰이며, 속성 단독 모델이 더 높게 나오면 그 결과를 README에 그대로 기록한다.
""")

nb.code("""
best_name = compare["Macro-F1"].idxmax()
print("주지표 최고 모델:", best_name)
print("서비스 탑재 모델: KLUE-RoBERTa 멀티태스크 (종합 평가 헤드 필요)")

SAVE_DIR = MODEL_DIR / "absa_klue"
meta = {"aspects": ASPECTS, "aspects_ko": ASPECTS_KO, "n_classes": 4,
        "label_names": LABEL_NAMES, "overall_names": OVERALL_NAMES,
        "max_len": CFG["max_len"], "base_model": CFG["model_name"], "cfg": CFG,
        "test_label_source": TEST_SOURCE, "test_metrics": {**summary(res_mt), **o}}
model_mt.save(SAVE_DIR, tokenizer, meta)

klue_metrics = {"model": "KLUE-RoBERTa multitask", **summary(res_mt), **o, "test_label_source": TEST_SOURCE,
                "per_aspect": res_mt.drop(index="MEAN")[["macro_f1", "accuracy", "mention_f1"]].astype(float).round(4).to_dict("index")}
if RUN_ABLATION:
    klue_metrics["ablation_aspect_only"] = summary(res_so)
if N_HUMAN:
    klue_metrics["human_subset_n"] = N_HUMAN
    klue_metrics["human_subset"] = hs_table.round(4).to_dict("index")
(RES_DIR / "klue_metrics.json").write_text(json.dumps(klue_metrics, ensure_ascii=False, indent=2))

pred = test[["recommendationid", "game", "text"]].copy()
for j, a in enumerate(ASPECTS):
    pred[f"{a}_true"], pred[f"{a}_pred"] = Y_te[:, j], P_te[:, j]
pred["overall_true"], pred["overall_pred"] = O_te, pred_mt[1]
pred["overall_p_rec"] = pred_mt[3][:, 1].round(4)
pred.to_csv(RES_DIR / "klue_test_pred.csv", index=False, encoding="utf-8-sig")

size = sum(p.stat().st_size for p in SAVE_DIR.iterdir()) / 1e6
print(f"저장: {SAVE_DIR} ({size:.0f} MB) — 100MB 초과이므로 GitHub 대신 HF Hub에 업로드")
""")

nb.code("""
# 저장한 파일만으로 다시 불러와 추론 — app.py와 같은 방식 (사전학습 가중치 재다운로드 없음)
del model_mt; torch.cuda.empty_cache()
m, tok, mt = AbsaModel.load(SAVE_DIR, device=DEVICE)

samples = ["그래픽은 정말 예쁜데 최적화가 엉망이라 프레임이 계속 떨어져요",
           "스토리랑 OST가 미쳤다. 가격도 할인하면 혜자",
           "서버 터지고 핵쟁이 천지. 운영 좀 제대로 해라",
           "소매치기만 하고 다니다 보니, 현실에서도 하고 싶어요.."]
pa, po = predict_proba(m, tok, [clean_text(s) for s in samples], max_len=mt["max_len"], device=DEVICE)
for s, a, p in zip(samples, pa.argmax(-1), po[:, 1]):
    asp = {ko: LABEL_NAMES[int(v)] for ko, v in zip(ASPECTS_KO, a) if v != 0}
    print(s); print(f"  → 종합: 추천 {p:.0%} | 속성: {asp or '언급 없음'}")
""")

nb.md("""
## 7. 배포

배포(HF 모델 업로드 + Spaces 앱)는 재학습 없이 CPU로 실행하는 **`06_deploy.ipynb`**에서 한다.
""")

nb.md("""
## 정리

- 모델 비교표(`results/model_comparison.csv`), 학습곡선·혼동행렬·속성별 비교 그래프는 README `## 4. 모델`, `## 5. 평가 결과`에 사용한다.
- 다음 단계: `06_deploy.ipynb` — HF Hub 업로드 + Spaces 배포
""")

nb.save(sys.argv[1], gpu=True)
