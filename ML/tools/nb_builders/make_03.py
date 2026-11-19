import sys
from nbtools import NB, SETUP

nb = NB()
nb.md("""
# 03. 라벨링 대상 샘플링 & 데이터 분할

정제 데이터(11,177건)에서 **2,000건**을 골라 속성별 감성 라벨을 붙이고, Train / Validation / Test로 나눈다.

**샘플링 전략**
- **무작위 풀 1,000건:** 실제 리뷰 분포를 그대로 반영 → Test 300 / Val 200 / Train 500
- **보강 풀 1,000건:** 무작위로는 거의 안 나오는 속성(사운드·조작감·그래픽 등)을 키워드로 찾아 추가 → **Train에만** 사용
- 보강 샘플을 Test에 넣지 않는 이유: 평가셋이 실제 분포와 달라지면 성능이 과대·과소 평가되기 때문

**Seed = 42** (샘플링·분할 모두)

라벨 기준은 `docs/labeling_guide.md` 참고.

**산출물:** `data/labeling/to_label.csv` (recommendationid, pool, split, text …)
""")

nb.code(SETUP)

nb.code("""
import re
import numpy as np
import pandas as pd

SEED = 42
N_RANDOM, N_BOOST = 1000, 1000
N_TEST, N_VAL = 300, 200

df = pd.read_csv(ROOT / "data" / "processed" / "reviews_clean.csv", dtype={"recommendationid": str})
print(f"정제 데이터: {len(df):,}건")
""")

nb.md("## 1. 무작위 풀 (1,000건) → Test / Val / Train 분할")

nb.code("""
random_pool = df.sample(n=N_RANDOM, random_state=SEED).copy()
random_pool["pool"] = "random"

# 무작위 풀 안에서 다시 섞어 Test 300 / Val 200 / Train 500
order = np.random.RandomState(SEED).permutation(len(random_pool))
split = np.array(["train"] * len(random_pool), dtype=object)
split[order[:N_TEST]] = "test"
split[order[N_TEST:N_TEST + N_VAL]] = "val"
random_pool["split"] = split
random_pool["split"].value_counts()
""")

nb.md("""
## 2. 보강 풀 (1,000건) — 희소 속성 키워드 샘플링

무작위 풀을 제외한 나머지에서, 속성별 키워드를 포함한 리뷰를 할당량만큼 뽑는다.
""")

nb.code("""
KEYWORDS = {
    "graphics":    r"그래픽|화질|비주얼|풍경|텍스처|아트|광원|모델링|graphic",
    "performance": r"최적화|프레임|fps|FPS|렉|랙|끊김|버벅|사양|로딩|발적화|그래픽카드",
    "story":       r"스토리|서사|엔딩|결말|등장인물|세계관|주인공|퀘스트",
    "controls":    r"조작|키보드|패드|컨트롤|타격감|UI|인터페이스|카메라|키 ?배치|마우스",
    "sound":       r"사운드|음악|bgm|BGM|OST|ost|브금|효과음|성우|더빙|음향|노래",
    "price":       r"가격|가성비|할인|세일|비싸|싸게|과금|현질|DLC|dlc|유료|돈값",
    "operation":   r"버그|튕|크래시|오류|핵쟁이|핵|서버|운영|패치|업데이트|개발사|한글화|번역",
}
QUOTA = {"sound": 180, "controls": 180, "graphics": 160, "performance": 140,
         "story": 120, "price": 120, "operation": 100}
assert sum(QUOTA.values()) == N_BOOST

rest = df[~df["recommendationid"].isin(random_pool["recommendationid"])]
picked, used = [], set()
for i, (aspect, quota) in enumerate(QUOTA.items()):
    cand = rest[rest["text"].str.contains(KEYWORDS[aspect], regex=True)
                & ~rest["recommendationid"].isin(used)]
    take = cand.sample(n=min(quota, len(cand)), random_state=SEED + i)
    used.update(take["recommendationid"])
    picked.append(take.assign(pool="boost", split="train", boost_aspect=aspect))
    print(f"{aspect:<12} 후보 {len(cand):>5}건 → {len(take)}건")

boost_pool = pd.concat(picked)
print("보강 풀:", len(boost_pool))
""")

nb.md("## 3. 합치기 및 저장")

nb.code("""
to_label = pd.concat([random_pool.assign(boost_aspect=""), boost_pool], ignore_index=True)
assert to_label["recommendationid"].is_unique

cols = ["recommendationid", "pool", "split", "boost_aspect", "game", "genre",
        "voted_up", "length", "text"]
to_label = to_label[cols]

OUT = ROOT / "data" / "labeling" / "to_label.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)
if OUT.exists():
    # 이미 라벨이 붙은 파일 → 덮어쓰지 않고 재현성만 확인 (행 순서가 바뀌면 라벨이 어긋나므로)
    prev = pd.read_csv(OUT, dtype={"recommendationid": str})
    same = prev["recommendationid"].tolist() == to_label["recommendationid"].tolist()
    print(f"기존 {OUT.name} 유지 ({len(prev)}건) | 같은 seed로 재샘플링한 결과와 동일: {same}")
    assert same, "샘플링 결과가 기존 파일과 다릅니다. 라벨과 어긋나므로 기존 파일을 사용하세요."
else:
    to_label.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("저장:", OUT, len(to_label))

summary = pd.crosstab(to_label["split"], to_label["pool"], margins=True)
summary
""")

nb.code("""
# 분할별 게임 분포·추천 비율 — Test가 원본 분포를 잘 대표하는지 확인
pd.DataFrame({
    "전체(정제)": df["voted_up"].mean(),
    "Train": to_label.query("split=='train'")["voted_up"].mean(),
    "Val":   to_label.query("split=='val'")["voted_up"].mean(),
    "Test":  to_label.query("split=='test'")["voted_up"].mean(),
}, index=["추천 비율"]).round(3)
""")

nb.md("""
## 다음 단계

`to_label.csv`의 2,000건에 `docs/labeling_guide.md` 기준으로 라벨을 붙인다 → `data/labeling/labels_llm.csv`
""")

nb.save(sys.argv[1])
