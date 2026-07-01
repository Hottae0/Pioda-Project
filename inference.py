"""
inference.py — 새 발화 데이터에 대한 이상 탐지 (재구성 오차 기반)

사용 예시:
    python inference.py

또는 다른 코드에서:
    from inference import AnomalyDetector
    detector = AnomalyDetector()
    score, is_anomaly = detector.score(features_list)
"""

import torch
import numpy as np
from model_train import VoiceLSTMAutoencoder


class AnomalyDetector:
    def __init__(
        self,
        model_path="voice_autoencoder.pt",
        norm_path="norm_stats.pt",
        threshold_path="threshold.pt",
    ):
        norm      = torch.load(norm_path, weights_only=True)
        self.mean = norm["mean"]
        self.std  = norm["std"]

        thresh          = torch.load(threshold_path, weights_only=True)
        self.threshold  = thresh["threshold"]

        self.model = VoiceLSTMAutoencoder(input_dim=4, hidden_dim=32, num_layers=2)
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def score(self, features: list[list[float]]) -> tuple[float, bool]:
        """
        features: [[speech_rate, ttr, repetition_rate, complexity], ...] (seq_length개)
        returns : (재구성 오차, 이상 여부)
        """
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, 4)
        x_norm = (x - self.mean) / (self.std + 1e-7)

        with torch.no_grad():
            recon = self.model(x_norm)
            error = ((recon - x_norm) ** 2).mean().item()

        is_anomaly = error > self.threshold
        return error, is_anomaly


# ── 간단한 테스트 ──────────────────────────────────────────

if __name__ == "__main__":
    detector = AnomalyDetector()

    # 정상 패턴 예시 (임의값 — 실제로는 파이프라인에서 추출된 피처)
    normal_seq = [
        [5.2, 0.75, 0.05, 12.0],
        [4.8, 0.70, 0.06, 11.5],
        [5.5, 0.78, 0.04, 13.0],
        [5.0, 0.72, 0.05, 12.2],
        [4.9, 0.71, 0.07, 11.8],
        [5.3, 0.76, 0.05, 12.5],
        [5.1, 0.74, 0.06, 12.1],
    ]

    # 이상 패턴 예시 (발화 속도 급감, TTR 하락, 반복 증가)
    anomaly_seq = [
        [2.1, 0.35, 0.40, 6.0],
        [1.8, 0.30, 0.45, 5.5],
        [2.3, 0.32, 0.42, 6.2],
        [1.9, 0.28, 0.50, 5.8],
        [2.0, 0.31, 0.48, 6.1],
        [1.7, 0.27, 0.55, 5.3],
        [2.2, 0.33, 0.43, 6.0],
    ]

    score_n, anomaly_n = detector.score(normal_seq)
    score_a, anomaly_a = detector.score(anomaly_seq)

    print(f"정상 패턴  — 오차: {score_n:.6f}  이상: {anomaly_n}")
    print(f"이상 패턴  — 오차: {score_a:.6f}  이상: {anomaly_a}")
    print(f"임계값: {detector.threshold:.6f}")
