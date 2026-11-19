"""라벨링용: to_label.csv의 a~b행을 한 줄씩 출력한다.

사용: python tools/show_batch.py 0 100 [최대글자수=500]
출력: 행번호|게임|U(추천)/D(비추천)|본문
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
P = Path(__file__).resolve().parents[1] / "data" / "labeling" / "to_label.csv"
df = pd.read_csv(P, dtype={"recommendationid": str})
a, b = int(sys.argv[1]), int(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 500
for i in range(a, min(b, len(df))):
    r = df.iloc[i]
    t = str(r["text"])[:limit].replace("\n", " ")
    print(f"{i}|{r['game'][:10]}|{'U' if r['voted_up'] else 'D'}|{t}")
