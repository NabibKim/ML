"""라벨 배치 파일(data/labeling/batches/*.txt)을 하나의 라벨 테이블로 합친다.

배치 파일 형식: 한 줄에 `<to_label 행 번호> <8자리 코드>`
코드 순서: graphics performance story gameplay controls sound price operation
값: 0=미언급 1=긍정 2=보통 3=부정

산출물
- data/labeling/labels_llm.csv   : 2,000건 전체 (LLM 라벨)
- data/labeling/test_verify.csv  : 테스트 300건 검수용 (사람이 수정 후 verified=1)
"""
import glob
import re
from pathlib import Path

import pandas as pd

ASPECTS = ["graphics", "performance", "story", "gameplay",
           "controls", "sound", "price", "operation"]
ASPECTS_KO = ["그래픽", "최적화", "스토리", "게임성", "조작감", "사운드", "가격", "운영"]
LABEL_NAMES = {0: "미언급", 1: "긍정", 2: "보통", 3: "부정"}


def load_codes(batch_dir: Path) -> dict:
    codes = {}
    for f in sorted(glob.glob(str(batch_dir / "b*.txt"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            idx, code = line.split()
            assert re.fullmatch(r"[0-3]{8}", code), (f, line)
            assert int(idx) not in codes, f"중복 행 번호: {idx}"
            codes[int(idx)] = code
    return codes


def main(root: Path):
    lab_dir = root / "data" / "labeling"
    to_label = pd.read_csv(lab_dir / "to_label.csv", dtype={"recommendationid": str})
    codes = load_codes(lab_dir / "batches")
    assert set(codes) == set(range(len(to_label))), "라벨 누락/초과"

    lab = pd.DataFrame([[int(c) for c in codes[i]] for i in range(len(to_label))],
                       columns=ASPECTS)
    out = pd.concat([to_label, lab], axis=1)
    out["label_source"] = "llm"
    out.to_csv(lab_dir / "labels_llm.csv", index=False, encoding="utf-8-sig")
    print(f"labels_llm.csv 저장: {len(out)}건")

    # 테스트셋 검수 파일 (엑셀에서 열기 쉽게 utf-8-sig)
    test = out[out["split"] == "test"].copy()
    verify = test[["recommendationid", "game", "voted_up", "text"] + ASPECTS].copy()
    verify.columns = (["recommendationid", "game", "voted_up", "text"]
                      + [f"{en}({ko})" for en, ko in zip(ASPECTS, ASPECTS_KO)])
    verify["verified"] = ""
    verify["memo"] = ""
    path = lab_dir / "test_verify.csv"
    if path.exists():
        print(f"⚠ {path.name} 이 이미 있어 덮어쓰지 않습니다 (검수 내용 보호).")
    else:
        verify.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"test_verify.csv 저장: {len(verify)}건")

    # 분포 요약
    dist = pd.DataFrame({a: out[a].value_counts().reindex(range(4), fill_value=0)
                         for a in ASPECTS}).T
    dist.columns = [LABEL_NAMES[i] for i in range(4)]
    dist["언급률"] = (1 - dist["미언급"] / len(out)).round(3)
    print(dist)


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
