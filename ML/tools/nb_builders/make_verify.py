import sys
from nbtools import NB, SETUP

nb = NB()
nb.md("""
# 테스트셋 검수 도구

`data/labeling/test_verify.csv`(테스트 300건)의 LLM 라벨을 사람이 확인·수정하는 노트북입니다.

**사용법**
1. 위에서부터 셀을 차례로 실행합니다.
2. **2번(목록 보기)** 으로 여러 건을 표로 훑어보고, **3번(검수 화면)** 에서 한 건씩 고칩니다.
3. 검수 화면에서 라벨을 고친 뒤 **[저장 후 다음]** 을 누르면 그 행이 `verified=1`로 표시되고 **즉시 CSV에 저장**됩니다. 중간에 꺼도 이어서 할 수 있습니다.
4. 처음 실행할 때 원본을 `test_verify_backup.csv`로 한 번 백업합니다.

라벨 기준은 `docs/labeling_guide.md` 참고 — 0 미언급 / 1 긍정 / 2 보통 / 3 부정
""")

nb.code(SETUP)

nb.code("""
import shutil
import pandas as pd
from IPython.display import display, HTML

PATH = ROOT / "data" / "labeling" / "test_verify.csv"
BACKUP = PATH.with_name("test_verify_backup.csv")
if not PATH.exists():
    raise FileNotFoundError(f"검수 파일이 없습니다: {PATH}\\n→ ROOT 경로와 data/labeling/test_verify.csv 위치를 확인하세요.")
if not BACKUP.exists():
    shutil.copy(PATH, BACKUP)
    print("원본 백업:", BACKUP.name)

ASPECT_COLS = ["graphics(그래픽)", "performance(최적화)", "story(스토리)", "gameplay(게임성)",
               "controls(조작감)", "sound(사운드)", "price(가격)", "operation(운영)"]
SHORT = [c.split("(")[1][:-1] for c in ASPECT_COLS]          # 그래픽, 최적화, ...
NAMES = {0: "미언급", 1: "긍정", 2: "보통", 3: "부정"}
COLORS = {0: "#9e9e9e", 1: "#2e7d32", 2: "#f9a825", 3: "#c62828"}

def load():
    d = pd.read_csv(PATH, dtype={"recommendationid": str})
    d["verified"] = pd.to_numeric(d["verified"], errors="coerce").fillna(0).astype(int)
    d["memo"] = d["memo"].fillna("").astype(str)
    for c in ASPECT_COLS:
        d[c] = d[c].astype(int)
    return d

def save(d):
    d.to_csv(PATH, index=False, encoding="utf-8-sig")

# 처음 상태(LLM 라벨) — 수정 여부 비교용
llm = pd.read_csv(BACKUP, dtype={"recommendationid": str})[ASPECT_COLS].astype(int)

df = load()
print(f"총 {len(df)}건 | 검수 완료 {df['verified'].sum()}건 | 남은 건 {(df['verified']==0).sum()}건")
""")

nb.md("""
## 1. 진행 현황
""")

nb.code("""
def progress():
    d = load()
    done = d["verified"] == 1
    changed = (d[ASPECT_COLS].values != llm.values)
    print(f"검수 완료 {done.sum()}/{len(d)}  ({done.mean():.0%})")
    print(f"수정된 행 {changed.any(axis=1).sum()}건 / 수정된 칸 {changed.sum()}개")
    by_aspect = pd.Series(changed.sum(axis=0), index=SHORT, name="수정 칸 수")
    display(by_aspect.to_frame().T)

progress()
""")

nb.md("""
## 2. 목록 보기 (표)

`START`, `N`을 바꿔가며 실행하세요. `ONLY_TODO=True`면 아직 검수하지 않은 행만 보여줍니다.
라벨은 색으로 구분됩니다: <span style="color:#2e7d32">긍정</span> / <span style="color:#f9a825">보통</span> / <span style="color:#c62828">부정</span> / <span style="color:#9e9e9e">미언급</span>
""")

nb.code("""
START, N = 0, 20
ONLY_TODO = False

def show_table(start=START, n=N, only_todo=ONLY_TODO):
    d = load()
    if only_todo:
        d = d[d["verified"] == 0]
    view = d.iloc[start:start + n].copy()
    view.insert(0, "번호", view.index)
    view["text"] = view["text"].str.slice(0, 500)
    view["voted_up"] = view["voted_up"].map({True: "👍", False: "👎", "True": "👍", "False": "👎"})
    show = view[["번호", "game", "voted_up", "text"] + ASPECT_COLS + ["verified", "memo"]]
    show = show.rename(columns=dict(zip(ASPECT_COLS, SHORT)))

    def color(v):
        return f"color: {COLORS[v]}; font-weight: {'bold' if v else 'normal'}"
    sty = show.style
    fmt = (sty.map if hasattr(sty, "map") else sty.applymap)
    sty = fmt(color, subset=SHORT).format({s: (lambda v: NAMES[v]) for s in SHORT})
    sty = sty.set_properties(subset=["text"], **{"white-space": "pre-wrap", "min-width": "420px",
                                                 "text-align": "left"})
    sty = sty.hide(axis="index")
    display(sty)

show_table()
""")

nb.md("""
## 3. 검수 화면 (한 건씩)

- 드롭다운으로 라벨을 고치고 **[저장 후 다음]** → 그 행이 `verified=1`로 저장됩니다.
- 라벨이 맞으면 아무것도 바꾸지 않고 **[저장 후 다음]** 만 누르면 됩니다.
- **[다음 미검수]** 는 아직 검수하지 않은 다음 행으로 건너뜁니다.
- 드롭다운 옆 `(LLM: 긍정)` 표시는 원래 라벨입니다. 바꾼 칸은 이름이 **주황색**으로 바뀝니다.
""")

nb.code("""
import ipywidgets as W

state = {"i": 0}
d0 = load()
todo = d0.index[d0["verified"] == 0]
state["i"] = int(todo[0]) if len(todo) else 0

header = W.HTML()
text_box = W.HTML(layout=W.Layout(border="1px solid #ccc", padding="8px", max_height="320px",
                                  overflow_y="auto", width="100%"))
options = [(f"{k} {v}", k) for k, v in NAMES.items()]
drops = [W.Dropdown(options=options, layout=W.Layout(width="130px")) for _ in ASPECT_COLS]
labels = [W.HTML() for _ in ASPECT_COLS]
memo = W.Text(placeholder="메모 (선택)", layout=W.Layout(width="100%"))
jump = W.BoundedIntText(min=0, max=len(d0) - 1, description="번호", layout=W.Layout(width="160px"))
msg = W.HTML()

b_prev = W.Button(description="◀ 이전")
b_next = W.Button(description="다음 ▶")
b_save = W.Button(description="저장 후 다음", button_style="success", icon="check")
b_todo = W.Button(description="다음 미검수 ⏭", button_style="info")
b_jump = W.Button(description="이동")

def label_html(j, val):
    orig = int(llm.iloc[state["i"], j])
    style = "color:#e65100;font-weight:bold" if val != orig else ""
    return f"<span style='{style}'>{SHORT[j]}</span> <small style='color:#888'>(LLM: {NAMES[orig]})</small>"

def render():
    d = load()
    i = state["i"]
    r = d.iloc[i]
    done = int(d["verified"].sum())
    mark = "✅ 검수됨" if r["verified"] == 1 else "⬜ 미검수"
    thumb = "👍 추천" if str(r["voted_up"]) == "True" else "👎 비추천"
    header.value = (f"<b>[{i}] {r['game']}</b> · {thumb} · {mark}"
                    f" &nbsp;&nbsp; <span style='color:#666'>진행 {done}/{len(d)}</span>")
    text_box.value = f"<div style='white-space:pre-wrap;font-size:15px;line-height:1.6'>{r['text']}</div>"
    for j, c in enumerate(ASPECT_COLS):
        drops[j].value = int(r[c])
        labels[j].value = label_html(j, int(r[c]))
    memo.value = r["memo"]
    jump.value = i

def on_drop_change(j):
    def f(change):
        labels[j].value = label_html(j, change["new"])
    return f
for j, dd in enumerate(drops):
    dd.observe(on_drop_change(j), names="value")

def move(k):
    state["i"] = max(0, min(len(load()) - 1, k))
    msg.value = ""
    render()

def on_save(_):
    d = load()
    i = state["i"]
    for j, c in enumerate(ASPECT_COLS):
        d.at[d.index[i], c] = int(drops[j].value)
    d.at[d.index[i], "verified"] = 1
    d.at[d.index[i], "memo"] = memo.value
    save(d)
    msg.value = f"<span style='color:green'>[{i}] 저장 완료</span>"
    if i < len(d) - 1:
        state["i"] = i + 1
    render()
    msg.value = f"<span style='color:green'>[{i}] 저장 완료</span>"

def on_todo(_):
    d = load()
    rest = [k for k in d.index[d["verified"] == 0] if k > state["i"]] or list(d.index[d["verified"] == 0])
    if rest:
        move(int(rest[0]))
    else:
        msg.value = "<b style='color:green'>🎉 300건 검수 완료!</b>"

b_prev.on_click(lambda _: move(state["i"] - 1))
b_next.on_click(lambda _: move(state["i"] + 1))
b_save.on_click(on_save)
b_todo.on_click(on_todo)
b_jump.on_click(lambda _: move(jump.value))

grid = W.GridBox(
    [W.HBox([labels[j], drops[j]], layout=W.Layout(justify_content="space-between")) for j in range(8)],
    layout=W.Layout(grid_template_columns="repeat(2, 1fr)", grid_gap="4px 24px", width="100%"))
ui = W.VBox([header, text_box, grid, memo,
             W.HBox([b_prev, b_next, b_save, b_todo, jump, b_jump]), msg])
render()
display(ui)
""")

nb.md("""
## 4. 수정 내역 확인

LLM 라벨과 달라진 칸만 모아 봅니다. 검수가 끝나면 이 표의 크기가 README `## 9. Lessons Learned`의 "LLM 라벨 정확도" 근거가 됩니다.
""")

nb.code("""
def diff_table():
    d = load()
    rows = []
    for i in range(len(d)):
        for j, c in enumerate(ASPECT_COLS):
            a, b = int(llm.iloc[i, j]), int(d.iloc[i][c])
            if a != b:
                rows.append({"번호": i, "속성": SHORT[j], "LLM": NAMES[a], "검수": NAMES[b],
                             "text": d.iloc[i]["text"][:120]})
    out = pd.DataFrame(rows)
    print(f"수정된 칸: {len(out)}개")
    return out

diff = diff_table()
diff.head(50)
""")

nb.code("""
# LLM 라벨 vs 사람 검수 일치도 (검수 완료 행 기준)
from sklearn.metrics import cohen_kappa_score

d = load()
done = d["verified"] == 1
if done.sum() == 0:
    print("아직 검수 완료된 행이 없습니다.")
else:
    rows = []
    for j, c in enumerate(ASPECT_COLS):
        a = llm.loc[done, c]
        b = d.loc[done, c]
        rows.append({"속성": SHORT[j], "일치율": (a == b).mean(),
                     "Cohen κ": cohen_kappa_score(a, b) if a.nunique() > 1 or b.nunique() > 1 else 1.0})
    agree = pd.DataFrame(rows).set_index("속성").round(3)
    print(f"검수 완료 {done.sum()}건 기준")
    display(agree)
    print(f"전체 칸 일치율: {(llm.loc[done].values == d.loc[done, ASPECT_COLS].values).mean():.3f}")
""")

nb.md("""
## 끝나면

- 검수 결과는 Drive의 `data/labeling/test_verify.csv`에 바로 저장되므로 따로 옮길 필요가 없습니다.
- 300행 모두 `verified=1`이면 이후 04·05 노트북이 자동으로 **사람 검수 라벨**로 평가합니다.
""")

nb.save(sys.argv[1])
