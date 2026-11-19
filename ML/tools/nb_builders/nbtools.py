import json
from pathlib import Path


class NB:
    def __init__(self):
        self.cells = []

    def md(self, s):
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})

    def code(self, s):
        self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                           "outputs": [], "source": s.strip("\n")})

    def save(self, out, gpu=False):
        meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}, "colab": {"provenance": []}}
        if gpu:
            meta["accelerator"] = "GPU"
            meta["colab"]["gpuType"] = "T4"
        for i, c in enumerate(self.cells):
            c["id"] = f"c{i:02d}"
            lines = c["source"].split("\n")
            c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        nb = {"cells": self.cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("wrote", out)


SETUP = '''
# ── 환경 설정 ────────────────────────────────────────────
import os, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")       # Drive 연결 위치(수정 X)

# ▶ 프로젝트 폴더 경로 (01 노트북과 동일하게)
def find_root():
    """프로젝트 폴더 자동 탐색. Drive의 한글 폴더명이 NFD(자모 분리)로 저장되면
    "ML 기말" 같은 문자열 경로가 일치하지 않으므로, MyDrive를 직접 훑어
    data/processed/reviews_clean.csv가 있는 폴더를 찾는다(검수 파일이 있는 쪽 우선)."""
    if os.environ.get("ML_ROOT"):
        return Path(os.environ["ML_ROOT"])
    base = Path("/content/drive/MyDrive")
    cands = [p for p in [base, *base.glob("*"), *base.glob("*/*")]
             if p.is_dir() and (p / "data" / "processed" / "reviews_clean.csv").exists()]
    if not cands:
        print("⚠ 프로젝트 폴더를 찾지 못해 기본 경로를 사용합니다.")
        return base / "ML 기말"
    cands.sort(key=lambda p: not (p / "data" / "labeling" / "test_verify.csv").exists())
    if len(cands) > 1:
        print("⚠ 프로젝트 폴더 후보가 여러 개입니다 → 첫 번째 사용:", *cands, sep="\\n   ")
    return cands[0]

ROOT = find_root()
sys.path.append(str(ROOT / "src"))      # src/preprocess.py 등 공용 모듈
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print("ROOT:", ROOT, "| exists:", ROOT.exists())
'''
