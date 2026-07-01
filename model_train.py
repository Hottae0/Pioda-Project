"""
model_train.py — 음성 지표 기반 LSTM 오토인코더 학습 스크립트

[개요]
    정상 노인 발화 데이터(AI Hub 자유대화)로 "정상 발화 패턴"을 학습한다.
    비지도 이상 탐지 방식 — 학습된 정상 패턴을 잘 복원하지 못하는(재구성 오차가 큰)
    입력을 '이상(인지 저하 의심)'으로 판단한다.

[입력 피처 4개]  (pipeline.extract_features 에서 추출)
    1) speech_rate      발화 속도   = 순수 음절 수 / 녹음 시간
    2) ttr              어휘 다양성 = 고유 내용어(명사·동사) / 전체 내용어
    3) repetition_rate  반복 표현   = 중복 2어절(bigram) 비율
    4) complexity       문장 복잡도 = 형태소 총 개수

[정규화]  Z-score. 학습 데이터 전체의 피처별 평균/표준편차로 (x-mean)/std.
          추론 시 동일 통계를 써야 하므로 norm_stats.pt 로 저장한다.

[산출물]
    voice_autoencoder.pt  학습된 모델 가중치
    norm_stats.pt         정규화 통계 (mean, std)  ← 추론 시 필수
    threshold.pt          이상 탐지 임계값 (mean + 2σ)

[실행]  python model_train.py   (데이터: aihub_data/[라벨]1.AI챗봇)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from pipeline import load_records_from_dir, build_sequences


# ── 모델 정의 ──────────────────────────────────────────────

class VoiceLSTMAutoencoder(nn.Module):
    """
    LSTM 오토인코더.
      encoder      : 시퀀스(seq_len, 4) → 마지막 hidden state로 압축
      decoder_lstm : 압축된 벡터를 seq_len 길이로 다시 펼쳐 복원
      output_layer : hidden_dim → 원래 피처 차원(4)로 매핑
    입력을 그대로 복원(reconstruct)하도록 학습하며, 복원 오차(MSE)를 이상 점수로 쓴다.
    """
    def __init__(self, input_dim=4, hidden_dim=32, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        # 인코더 마지막 층의 hidden state를 시퀀스 길이만큼 복제해 디코더 입력으로
        hidden_last = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder_lstm(hidden_last)
        return self.output_layer(decoded)


# ── 데이터 로드 ────────────────────────────────────────────

DATA_DIR   = "aihub_data/[라벨]1.AI챗봇"
SEQ_LENGTH = 7

records = load_records_from_dir(DATA_DIR)
voice_data = build_sequences(records, seq_length=SEQ_LENGTH)

if voice_data is None:
    raise RuntimeError("시퀀스 생성 실패. 데이터를 확인하세요.")


# ── 정규화 ─────────────────────────────────────────────────

mean = voice_data.mean(dim=(0, 1), keepdim=True)
std  = voice_data.std(dim=(0, 1), keepdim=True)
voice_data_norm = (voice_data - mean) / (std + 1e-7)

# 정규화 통계 저장 (추론 시 필요)
torch.save({"mean": mean, "std": std}, "norm_stats.pt")
print(f"[train] 정규화 통계 저장 완료")


# ── 학습 ───────────────────────────────────────────────────

dataset    = TensorDataset(voice_data_norm, voice_data_norm)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model     = VoiceLSTMAutoencoder(input_dim=4, hidden_dim=32, num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 100
print(f"[train] 학습 시작 ({EPOCHS} epochs, {len(dataloader)} batches/epoch)")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = epoch_loss / len(dataloader)
        print(f"  Epoch [{epoch+1}/{EPOCHS}]  Loss: {avg_loss:.6f}")


# ── 학습 오차 분포 계산 (이상 탐지 임계값 기준) ───────────

model.eval()
with torch.no_grad():
    all_errors = []
    for (batch_x,) in DataLoader(TensorDataset(voice_data_norm), batch_size=256):
        recon = model(batch_x)
        errors = ((recon - batch_x) ** 2).mean(dim=(1, 2))
        all_errors.append(errors)

    all_errors = torch.cat(all_errors)
    threshold_mean = all_errors.mean().item()
    threshold_std  = all_errors.std().item()
    threshold = threshold_mean + 2 * threshold_std  # 정상 범위 상한

print(f"\n[train] 재구성 오차 — mean: {threshold_mean:.6f}, std: {threshold_std:.6f}")
print(f"[train] 이상 탐지 임계값 (mean + 2σ): {threshold:.6f}")


# ── 저장 ───────────────────────────────────────────────────

torch.save(model.state_dict(), "voice_autoencoder.pt")
torch.save({"threshold": threshold, "mean": threshold_mean, "std": threshold_std},
           "threshold.pt")

print("\n[train] 저장 완료:")
print("  voice_autoencoder.pt  — 모델 가중치")
print("  norm_stats.pt         — 정규화 통계 (mean/std)")
print("  threshold.pt          — 이상 탐지 임계값")
