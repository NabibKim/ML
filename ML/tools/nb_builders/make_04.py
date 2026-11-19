import sys
from nbtools import NB, SETUP

nb = NB()
nb.md("""
# 04. 베이스라인 — TF-IDF + 로지스틱 회귀

**목표:** 속성 8개 × {0 미언급, 1 긍정, 2 보통, 3 부정}을 예측하는 **고전 ML 베이스라인**을 학습하고, 05의 제안 모델(KLUE-RoBERTa)과 비교할 기준점을 만든다.

| 항목 | 내용 |
|---|---|
| 입력 | 정제된 리뷰 텍스트(`src/preprocess.clean_text` 적용본) |
| 특징 | TF-IDF **글자 n-gram(2~4, 단어 경계 기준)** — 한국어는 조사·어미가 붙어 단어 단위가 잘게 깨지므로 형태소 분석기 없이도 강건한 글자 n-gram을 사용 |
| 분류기 | 속성별 **로지스틱 회귀 8개** (다중 클래스, `class_weight="balanced"`로 미언급 쏠림 완화) |
| 튜닝 | 속성마다 규제 강도 `C`를 **Validation Macro-F1** 기준으로 선택 |
| 학습 방식 | Scratch (사전학습 없음) |

**평가 지표** (`src/absa.py` — 05와 동일한 함수 사용)
- **주지표: 속성 평균 Macro-F1** — 속성별 4클래스 Macro-F1의 평균. 정답·예측 어디에도 없는 클래스는 평균에서 제외
- 보조: Accuracy, **언급 감지 F1**(미언급 vs 언급) — 사운드처럼 테스트 언급이 매우 적은 속성은 Macro-F1 변동이 커서 함께 보고

**종합 평가(overall):** 속성이 모두 미언급인 은유적·총평형 리뷰("소매치기만 하고 다니다 보니 현실에서도 하고 싶어요..")를 위해 **추천/비추천**을 따로 예측한다. 정답은 작성자가 누른 추천 버튼(`voted_up`)이라 추가 라벨링 없이 정제 데이터 전체(Test·Val 제외 10,677건)로 학습한다.

**산출물:** `models/baseline.joblib`, `results/baseline_metrics.json`, `results/baseline_test_pred.csv`, `figures/04_*.png`
""")

nb.code(SETUP)

nb.code("""
import json, time
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from absa import (ASPECTS, ASPECTS_KO, LABELS, LABEL_NAMES, LABEL_NAMES_EN, OVERALL_NAMES,
                  load_splits, load_overall_splits, evaluate, summary, aspect_macro_f1, overall_metrics,
                  human_subset_report)
from preprocess import clean_text

SEED = 42
MODEL_DIR = ROOT / "models"; MODEL_DIR.mkdir(exist_ok=True)
RES_DIR = ROOT / "results"; RES_DIR.mkdir(exist_ok=True)

train, val, test, TEST_SOURCE = load_splits(ROOT)
Y_tr, Y_va, Y_te = (d[ASPECTS].values.astype(int) for d in (train, val, test))
""")

nb.md("""
## 1. 분할별 클래스 분포

Train에는 희소 속성 보강 풀(1,000건)이 포함되어 있고, Val·Test는 무작위 풀에서만 뽑아 **실제 분포**를 따른다.
""")

nb.code("""
def dist_table(Y, name):
    t = pd.DataFrame({ko: pd.Series(Y[:, j]).value_counts().reindex(LABELS, fill_value=0)
                      for j, ko in enumerate(ASPECTS_KO)}).T
    t.columns = [LABEL_NAMES[c] for c in LABELS]
    t.columns = pd.MultiIndex.from_product([[name], t.columns])
    return t

pd.concat([dist_table(Y_tr, f"Train ({len(train)})"),
           dist_table(Y_va, f"Val ({len(val)})"),
           dist_table(Y_te, f"Test ({len(test)})")], axis=1)
""")

nb.md("## 2. 특징 추출 — TF-IDF 글자 n-gram (Train으로만 fit)")

nb.code("""
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                             max_features=50000, sublinear_tf=True)
t0 = time.time()
X_tr = vectorizer.fit_transform(train["text"])
X_va = vectorizer.transform(val["text"])
X_te = vectorizer.transform(test["text"])
print(f"어휘 수 {len(vectorizer.vocabulary_):,} | X_train {X_tr.shape} | {time.time()-t0:.1f}s")
""")

nb.md("""
## 3. 학습 및 하이퍼파라미터 튜닝 (학습 로그)

속성마다 `C ∈ {0.1, 0.3, 1, 3, 10, 30}`을 시도하고 **Validation Macro-F1**이 가장 높은 값을 고른다. 아래 표와 그래프가 베이스라인의 학습 로그다.
""")

nb.code("""
C_GRID = [0.1, 0.3, 1, 3, 10, 30]
log_rows, best = [], {}

for j, a in enumerate(ASPECTS):
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
        clf.fit(X_tr, Y_tr[:, j])
        tr_f1 = aspect_macro_f1(Y_tr[:, j], clf.predict(X_tr))
        va_f1 = aspect_macro_f1(Y_va[:, j], clf.predict(X_va))
        log_rows.append({"aspect": a, "C": C, "train_macro_f1": tr_f1, "val_macro_f1": va_f1})
        if a not in best or va_f1 > best[a][1]:
            best[a] = (C, va_f1, clf)
    print(f"{a:<12} best C={best[a][0]:<5} val Macro-F1={best[a][1]:.3f}")

train_log = pd.DataFrame(log_rows)
train_log.to_csv(RES_DIR / "baseline_train_log.csv", index=False)
clfs = {a: best[a][2] for a in ASPECTS}
BEST_C = {a: best[a][0] for a in ASPECTS}
train_log.pivot(index="aspect", columns="C", values="val_macro_f1").reindex(ASPECTS).round(3)
""")

nb.code("""
# C에 따른 Train / Val Macro-F1 — 규제가 약해질수록(C↑) Train은 오르고 Val은 꺾이는 과적합 구간 확인
fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True, sharey=True)
for ax, a in zip(axes.ravel(), ASPECTS):
    d = train_log[train_log["aspect"] == a]
    ax.plot(d["C"], d["train_macro_f1"], "o-", label="train")
    ax.plot(d["C"], d["val_macro_f1"], "o-", label="val")
    ax.axvline(BEST_C[a], color="gray", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_title(a); ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
axes[0, 0].legend()
fig.supxlabel("C (inverse regularization, log scale)"); fig.supylabel("Macro-F1")
fig.suptitle("Baseline (TF-IDF + LR): Macro-F1 vs C")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_baseline_tuning_curve.png", dpi=150, bbox_inches="tight")
plt.show()
""")

nb.md(f"""
## 4. 테스트셋 평가

테스트셋은 튜닝에 한 번도 쓰지 않았다. 테스트 라벨은 학습 데이터와 같은 기준인 **LLM 라벨**이다.
""")

nb.code("""
P_te = np.column_stack([clfs[a].predict(X_te) for a in ASPECTS])
res = evaluate(Y_te, P_te)
print(f"테스트 라벨 출처: {'사람 검수' if TEST_SOURCE == 'human' else 'LLM'}")
print("요약:", {k: round(v, 4) for k, v in summary(res).items()})
res.round(3)
""")

nb.code("""
# 속성 × 클래스별 F1 — 어떤 클래스가 약한지(특히 '보통') 확인
rows = []
for j, a in enumerate(ASPECTS):
    f = f1_score(Y_te[:, j], P_te[:, j], labels=LABELS, average=None, zero_division=0)
    sup = [(Y_te[:, j] == c).sum() for c in LABELS]
    rows.append({**{f"F1-{LABEL_NAMES[c]}": f[i] for i, c in enumerate(LABELS)},
                 **{f"n-{LABEL_NAMES[c]}": sup[i] for i, c in enumerate(LABELS)}})
pd.DataFrame(rows, index=ASPECTS_KO).round(3)
""")

nb.code("""
# 혼동행렬 (행: 정답, 열: 예측)
fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
ticks = [LABEL_NAMES_EN[c] for c in LABELS]
for j, (ax, a) in enumerate(zip(axes.ravel(), ASPECTS)):
    cm = confusion_matrix(Y_te[:, j], P_te[:, j], labels=LABELS)
    ax.imshow(cm, cmap="Blues")
    for r in range(4):
        for c in range(4):
            ax.text(c, r, cm[r, c], ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks(range(4), ticks); ax.set_yticks(range(4), ticks)
    ax.set_title(f"{a}  (F1={res.loc[a, 'macro_f1']:.2f})")
    ax.set_xlabel("pred"); ax.set_ylabel("true")
fig.suptitle("Baseline confusion matrices (test)")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_baseline_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
""")

nb.md("""
### [참고] 사람 검수 시도와의 비교

테스트셋을 사람이 검수해 보았으나(약 150건 정독, 이후는 훑어봄) 경계 사례의 기준이 일정하지 않아 **최종 평가는 LLM 라벨로 통일**했다. 아래는 검수 라벨과 LLM 라벨이 얼마나 달랐는지(일치율·Cohen κ), 검수 라벨로 채점하면 점수가 얼마나 바뀌는지를 참고로 보여준다.
""")

nb.code("""
hs_table, hs_agree, N_HUMAN = human_subset_report(ROOT, test, {"Baseline: TF-IDF + LR": P_te})
if N_HUMAN:
    display(hs_agree.round(3))
    display(hs_table.round(4))
""")

nb.md("""
## 5. 오류 사례

정답과 예측이 다른 칸을 모아 몇 건을 직접 읽어 본다 → README `## 5. 평가 결과` 분석과 Lessons Learned 소재.
""")

nb.code("""
err = []
for j, a in enumerate(ASPECTS):
    for i in np.where(Y_te[:, j] != P_te[:, j])[0]:
        err.append({"idx": i, "속성": ASPECTS_KO[j], "정답": LABEL_NAMES[Y_te[i, j]],
                    "예측": LABEL_NAMES[P_te[i, j]], "text": test.loc[i, "text"][:150]})
err = pd.DataFrame(err)
print(f"틀린 칸 {len(err)}개 / 전체 {Y_te.size}칸")
print(pd.crosstab(err["정답"], err["예측"]))     # 어떤 방향으로 틀리는지
pd.set_option("display.max_colwidth", 150)
err.sample(min(12, len(err)), random_state=SEED)
""")

nb.md("""
## 6. 종합 평가(추천/비추천) — 별도 로지스틱 회귀

- 학습: 정제 데이터 중 Test·Val을 제외한 10,677건 (라벨 = `voted_up`)
- 평가: 속성과 **같은** Test 300건 / Val 200건
- 추천이 약 83%로 불균형 → `class_weight="balanced"`, 주지표 Macro-F1
""")

nb.code("""
o_tr, o_va, o_te = load_overall_splits(ROOT)
vec_o = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=100000, sublinear_tf=True)
Xo_tr = vec_o.fit_transform(o_tr["text"]); Xo_va = vec_o.transform(o_va["text"]); Xo_te = vec_o.transform(o_te["text"])

o_log, o_best = [], None
for C in C_GRID:
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    clf.fit(Xo_tr, o_tr["overall"])
    va_f1 = f1_score(o_va["overall"], clf.predict(Xo_va), average="macro")
    o_log.append({"C": C, "val_macro_f1": round(va_f1, 4)})
    if o_best is None or va_f1 > o_best[1]:
        o_best = (C, va_f1, clf)
clf_o = o_best[2]
print(pd.DataFrame(o_log).to_string(index=False))
print(f"best C={o_best[0]}")

Po_te = clf_o.predict(Xo_te)
overall_res = overall_metrics(o_te["overall"], Po_te)
print("테스트:", {k: round(v, 4) for k, v in overall_res.items()})
pd.DataFrame(confusion_matrix(o_te["overall"], Po_te, labels=[0, 1]),
             index=["정답 비추천", "정답 추천"], columns=["예측 비추천", "예측 추천"])
""")

nb.code("""
# 속성이 전부 '미언급'으로 예측된 리뷰 — 종합 평가가 이 리뷰들을 어떻게 받아주는지 확인
all_none = (P_te == 0).all(axis=1)
print(f"속성 전부 미언급 예측: {all_none.sum()}건 / {len(test)}건")
demo = test.loc[all_none, ["text"]].copy()
demo["종합(예측)"] = [OVERALL_NAMES[int(v)] for v in Po_te[all_none]]
demo["종합(정답)"] = [OVERALL_NAMES[int(v)] for v in o_te["overall"].values[all_none]]
demo["text"] = demo["text"].str.slice(0, 80)
demo.head(10)
""")

nb.md("## 7. 저장 및 로드 확인 (서비스와 같은 방식으로 추론)")

nb.code("""
bundle = {"vectorizer": vectorizer, "clfs": clfs, "best_C": BEST_C, "aspects": ASPECTS,
          "overall_vectorizer": vec_o, "overall_clf": clf_o}
joblib.dump(bundle, MODEL_DIR / "baseline.joblib", compress=3)

metrics = {"model": "TF-IDF(char 2-4) + LogisticRegression", "test_label_source": TEST_SOURCE,
           **summary(res), "per_aspect": res.drop(index="MEAN")[["macro_f1", "accuracy", "mention_f1"]].round(4).to_dict("index"),
           "best_C": BEST_C, **overall_res, "overall_best_C": o_best[0],
           "human_subset_n": N_HUMAN,
           "human_subset": (hs_table.loc["Baseline: TF-IDF + LR"].to_dict() if N_HUMAN else None)}
(RES_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

pred = test[["recommendationid", "game", "text"]].copy()
for j, a in enumerate(ASPECTS):
    pred[f"{a}_true"], pred[f"{a}_pred"] = Y_te[:, j], P_te[:, j]
pred.to_csv(RES_DIR / "baseline_test_pred.csv", index=False, encoding="utf-8-sig")

size_mb = (MODEL_DIR / "baseline.joblib").stat().st_size / 1e6
print(f"저장 완료: models/baseline.joblib ({size_mb:.1f} MB), results/baseline_metrics.json, results/baseline_test_pred.csv")
""")

nb.code("""
# 저장한 파일을 다시 불러와 새 문장 추론 — app.py에서 쓰는 방식 그대로
m = joblib.load(MODEL_DIR / "baseline.joblib")

def predict(text):
    x = m["vectorizer"].transform([clean_text(text)])
    out = {ko: LABEL_NAMES[int(m["clfs"][a].predict(x)[0])] for a, ko in zip(m["aspects"], ASPECTS_KO)}
    xo = m["overall_vectorizer"].transform([clean_text(text)])
    p_rec = m["overall_clf"].predict_proba(xo)[0, 1]
    return out, p_rec

for s in ["그래픽은 정말 예쁜데 최적화가 엉망이라 프레임이 계속 떨어져요",
          "스토리랑 OST가 미쳤다. 가격도 할인하면 혜자",
          "서버 터지고 핵쟁이 천지. 운영 좀 제대로 해라",
          "소매치기만 하고 다니다 보니, 현실에서도 하고 싶어요.."]:
    asp, p = predict(s)
    print(s); print(f"  → 종합: 추천 {p:.0%} | 속성:", {k: v for k, v in asp.items() if v != "미언급"} or "언급 없음")
""")

nb.md("""
## 정리

- 베이스라인 테스트 성능은 `results/baseline_metrics.json`에 저장되어 05 노트북의 **모델 비교표**에서 그대로 불러온다.
- 다음 단계: `05_transformer.ipynb` — KLUE-RoBERTa 멀티헤드 파인튜닝
""")

nb.save(sys.argv[1])
