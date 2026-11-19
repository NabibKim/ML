"""ABSA 공용 모듈 — 데이터 로딩 · 평가 함수.

04(베이스라인)·05(제안 모델)·app이 **같은 분할, 같은 지표 계산**을 쓰도록 한곳에 둔다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ASPECTS = ["graphics", "performance", "story", "gameplay",
           "controls", "sound", "price", "operation"]
ASPECTS_KO = ["그래픽", "최적화", "스토리", "게임성", "조작감", "사운드", "가격", "운영"]
LABELS = [0, 1, 2, 3]
LABEL_NAMES = {0: "미언급", 1: "긍정", 2: "보통", 3: "부정"}
LABEL_NAMES_EN = {0: "none", 1: "pos", 2: "neu", 3: "neg"}


def load_splits(root: Path, verbose: bool = True, use_human_test: bool = False):
    """labels_llm.csv를 split 컬럼으로 나눈다. **테스트 라벨은 기본적으로 LLM 라벨**을 쓴다.

    사람 검수(test_verify.csv)는 기준이 일정하지 않아(약 150건 정독, 이후 훑어봄) 최종 평가에 쓰지 않고,
    `human_subset_report`로 LLM 라벨과의 차이를 참고용으로만 보고한다.
    use_human_test=True로 하면 검수 완료 행의 라벨로 테스트 라벨을 교체한다(비교 실험용).
    반환: train, val, test DataFrame, test_label_source ('llm' | 'human')
    """
    lab_dir = Path(root) / "data" / "labeling"
    df = pd.read_csv(lab_dir / "labels_llm.csv", dtype={"recommendationid": str})
    df["text"] = df["text"].fillna("").astype(str)

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)

    source = "llm"
    if use_human_test:
        human = load_verified(root)
        if len(human):
            m = test["recommendationid"].isin(human.index)
            test.loc[m, ASPECTS] = human.loc[test.loc[m, "recommendationid"]].values
            source = "human"
    test["label_source"] = source

    if verbose:
        print(f"Train {len(train)} / Val {len(val)} / Test {len(test)}  "
              f"(테스트 라벨: {'사람 검수' if source == 'human' else 'LLM — 사람 검수는 참고용으로만 비교'})")
    return train, val, test, source


def load_verified(root: Path) -> pd.DataFrame:
    """test_verify.csv에서 사람이 검수한(verified=1) 행만 → index=recommendationid, 컬럼=ASPECTS."""
    vpath = Path(root) / "data" / "labeling" / "test_verify.csv"
    if not vpath.exists():
        return pd.DataFrame(columns=ASPECTS)
    v = pd.read_csv(vpath, dtype={"recommendationid": str})
    done = pd.to_numeric(v["verified"], errors="coerce").fillna(0).astype(int) == 1
    col_map = {f"{en}({ko})": en for en, ko in zip(ASPECTS, ASPECTS_KO)}
    return v[done].rename(columns=col_map).set_index("recommendationid")[ASPECTS].astype(int)


def human_subset_report(root: Path, test: pd.DataFrame, preds: dict, verbose: bool = True):
    """사람 검수 부분집합 평가 (검수를 일부만 한 경우의 보조 평가).

    preds: {"모델 이름": (N_test, 8) 예측 배열} — test 행 순서와 같아야 함
    반환: (모델별 지표 DataFrame, LLM-사람 일치도 DataFrame, 검수 건수)
    """
    from sklearn.metrics import cohen_kappa_score
    human = load_verified(root)
    n = len(human)
    if n == 0:
        if verbose:
            print("사람 검수 완료 행이 없습니다 — 보조 평가를 건너뜁니다.")
        return None, None, 0
    lab = pd.read_csv(Path(root) / "data" / "labeling" / "labels_llm.csv", dtype={"recommendationid": str})
    llm = lab.set_index("recommendationid").loc[human.index, ASPECTS].astype(int)
    mask = test["recommendationid"].isin(human.index).values
    ids = test.loc[mask, "recommendationid"]
    Y_h = human.loc[ids].values

    agree = pd.DataFrame({
        "속성": ASPECTS_KO,
        "일치율": [(llm[a] == human[a]).mean() for a in ASPECTS],
        "Cohen κ": [cohen_kappa_score(llm[a], human[a]) if (llm[a].nunique() > 1 or human[a].nunique() > 1) else 1.0
                    for a in ASPECTS],
        "수정 칸 수": [int((llm[a] != human[a]).sum()) for a in ASPECTS],
    }, index=ASPECTS)
    cell_agree = float((llm.values == human.values).mean())
    row_changed = int((llm.values != human.values).any(axis=1).sum())

    rows = {"LLM 라벨 (검수 라벨 대비)": summary(evaluate(Y_h, llm.loc[ids].values))}
    for name, P in preds.items():
        rows[name] = summary(evaluate(Y_h, np.asarray(P)[mask]))
    table = pd.DataFrame(rows).T
    if verbose:
        print(f"[참고] 검수 시도 {n}행 (기준이 일정하지 않아 최종 평가에는 미사용) | LLM-검수 칸 일치율 {cell_agree:.3f} | "
              f"검수에서 라벨이 바뀐 리뷰 {row_changed}건 ({row_changed / n:.0%})")
    return table, agree, n


OVERALL_NAMES = {0: "비추천", 1: "추천"}


def load_overall_splits(root: Path, verbose: bool = True):
    """종합 평가(voted_up: 작성자의 추천 여부) 데이터.

    라벨은 리뷰 작성자가 직접 누른 추천 버튼이므로 추가 라벨링이 필요 없다.
    - Test / Val: 속성 데이터와 **같은 리뷰**(300 / 200건) → 두 과제를 같은 테스트셋에서 평가
    - Train: 정제 데이터 11,177건 중 Test·Val을 뺀 전부 (속성 Train 1,500건 포함)
    반환: train, val, test DataFrame (컬럼 text, overall ∈ {0,1})
    """
    root = Path(root)
    clean = pd.read_csv(root / "data" / "processed" / "reviews_clean.csv", dtype={"recommendationid": str})
    lab = pd.read_csv(root / "data" / "labeling" / "labels_llm.csv", dtype={"recommendationid": str})
    clean["text"] = clean["text"].fillna("").astype(str)
    clean["overall"] = clean["voted_up"].astype(str).str.lower().eq("true").astype(int)

    test_ids = lab.loc[lab["split"] == "test", "recommendationid"]
    val_ids = lab.loc[lab["split"] == "val", "recommendationid"]
    by_id = clean.set_index("recommendationid")
    test = by_id.loc[test_ids].reset_index()
    val = by_id.loc[val_ids].reset_index()
    train = clean[~clean["recommendationid"].isin(set(test_ids) | set(val_ids))].reset_index(drop=True)
    if verbose:
        print(f"[종합 평가] Train {len(train):,} / Val {len(val)} / Test {len(test)} | "
              f"추천 비율 Train {train['overall'].mean():.3f}, Test {test['overall'].mean():.3f}")
    return train, val, test


def overall_metrics(y_true, y_pred) -> dict:
    """종합 평가 지표 — 추천이 약 83%로 불균형이라 Macro-F1을 주지표로 본다."""
    return {"overall_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "overall_accuracy": float(accuracy_score(y_true, y_pred)),
            "overall_f1_neg": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))}


def aspect_macro_f1(y_true, y_pred) -> float:
    """4클래스 Macro-F1. 정답·예측 어디에도 없는 클래스는 평균에서 제외한다
    (테스트에 '사운드-보통'이 0건이면 그 클래스를 0점으로 세지 않기 위해)."""
    present = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    return f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)


def mention_f1(y_true, y_pred) -> float:
    """언급 감지 F1: 미언급(0) vs 언급(1~3) 이진 분류."""
    t = (np.asarray(y_true) > 0).astype(int)
    p = (np.asarray(y_pred) > 0).astype(int)
    return f1_score(t, p, zero_division=0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """(N, 8) 정답·예측 → 속성별 지표 표 + 평균 행."""
    rows = []
    for j, a in enumerate(ASPECTS):
        t, p = y_true[:, j], y_pred[:, j]
        rows.append({
            "aspect": a,
            "속성": ASPECTS_KO[j],
            "macro_f1": aspect_macro_f1(t, p),
            "accuracy": accuracy_score(t, p),
            "mention_f1": mention_f1(t, p),
            "n_mentioned": int((t > 0).sum()),
        })
    res = pd.DataFrame(rows).set_index("aspect")
    res.loc["MEAN"] = ["평균", res["macro_f1"].mean(), res["accuracy"].mean(),
                       res["mention_f1"].mean(), int(res["n_mentioned"].sum())]
    return res


def summary(res: pd.DataFrame) -> dict:
    """비교표용 요약 지표."""
    m = res.loc["MEAN"]
    return {"macro_f1": float(m["macro_f1"]), "accuracy": float(m["accuracy"]),
            "mention_f1": float(m["mention_f1"])}
