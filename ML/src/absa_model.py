"""KLUE-RoBERTa 멀티헤드 ABSA 모델 — 학습(05 노트북)과 서비스(app.py)가 같은 클래스를 쓴다.

구조: 공유 인코더([CLS] 벡터) → 속성 헤드 8개(각 4클래스) + 종합 평가 헤드 1개(추천/비추천)
저장 형식(models/absa_klue/):
  config.json, tokenizer 파일들   ← 인코더 구조·토크나이저 (사전학습 가중치 재다운로드 불필요)
  model.pt                        ← 전체 state_dict (fp16으로 저장해 용량 절반)
  absa_meta.json                  ← 속성 목록·클래스 수·최대 길이·학습 정보
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class AbsaModel(nn.Module):
    def __init__(self, encoder, n_aspects=8, n_classes=4, n_overall=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.aspect_heads = nn.ModuleList([nn.Linear(hidden, n_classes) for _ in range(n_aspects)])
        self.overall_head = nn.Linear(hidden, n_overall)

    @classmethod
    def from_pretrained_encoder(cls, name, **kw):
        return cls(AutoModel.from_pretrained(name), **kw)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = self.dropout(out.last_hidden_state[:, 0])
        aspect_logits = torch.stack([h(cls_vec) for h in self.aspect_heads], dim=1)  # (B, A, C)
        overall_logits = self.overall_head(cls_vec)                                  # (B, 2)
        return aspect_logits, overall_logits

    def save(self, out_dir, tokenizer, meta: dict):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        state = {k: (v.half() if v.is_floating_point() else v) for k, v in self.state_dict().items()}
        torch.save(state, out_dir / "model.pt")
        (out_dir / "absa_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir, device="cpu"):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "absa_meta.json").read_text(encoding="utf-8"))
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_dir))
        model = cls(encoder, n_aspects=len(meta["aspects"]), n_classes=meta["n_classes"])
        state = torch.load(model_dir / "model.pt", map_location="cpu")
        model.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in state.items()})
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        return model.to(device).eval(), tokenizer, meta


@torch.no_grad()
def predict_proba(model, tokenizer, texts, max_len=256, batch_size=32, device="cpu"):
    """texts(list[str], 이미 clean_text 적용) → (속성 확률 (N, A, C), 종합 확률 (N, 2)) numpy."""
    model.eval()
    asp, ovr = [], []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[i:i + batch_size]), padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        a, o = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        asp.append(torch.softmax(a.float(), -1).cpu().numpy())
        ovr.append(torch.softmax(o.float(), -1).cpu().numpy())
    return np.concatenate(asp), np.concatenate(ovr)
