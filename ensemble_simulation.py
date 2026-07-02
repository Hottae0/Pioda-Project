import torch
import torch.nn as nn
import numpy as np


# ==========================================
# 1. 모델 아키텍처 정의 (생략 - 기존과 동일)
# ==========================================
class VoiceLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden_last = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder_lstm(hidden_last)
        return self.output_layer(decoded)


class VisionLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=32, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        hidden_last = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder_lstm(hidden_last)
        return self.output_layer(decoded)


# ==========================================
# 2. 모델 로드 및 설정
# ==========================================
print("멀티모달 가중치 및 환경 파일 로드 중...")
voice_model = VoiceLSTMAutoencoder(input_dim=4, hidden_dim=16, num_layers=1)
vision_model = VisionLSTMAutoencoder(input_dim=20, hidden_dim=32, num_layers=2)

try:
    voice_model.load_state_dict(torch.load('voice_lstm_model.pth', map_location=torch.device('cpu')))
    vision_model.load_state_dict(torch.load('vision_autoencoder.pt', map_location=torch.device('cpu')))
    vision_norm_stats = torch.load('norm_stats.pt', map_location=torch.device('cpu'))
    vision_threshold_data = torch.load('threshold.pt', map_location=torch.device('cpu'))

    vision_mean = vision_norm_stats['mean']
    vision_std = vision_norm_stats['std']

    # [방법 1] 비전 단독 임계값을 3-sigma 혹은 4-sigma로 확장하여 보수적으로 잡기
    t_mean = vision_threshold_data['mean']
    t_std = vision_threshold_data['std']

    # 기존 2-sigma(1.145)에서 4-sigma로 변경하여 엄격한 정상 기준 수립
    adjusted_vision_threshold = t_mean + 4 * t_std
    print("모델 및 산출물 로드 완료!")
    print(f"[방법 1] 조정된 비전 단독 임계값 (Mean + 4σ): {adjusted_vision_threshold:.6f}")
except FileNotFoundError as e:
    print(f"파일 로드 실패: {e}")
    exit()

voice_model.eval()
vision_model.eval()
criterion = nn.MSELoss(reduction='none')


# ==========================================
# 3. 데이터 생성 함수 (기존과 동일)
# ==========================================
def generate_ensemble_test_data(num_samples, v_mean, v_std, scenario='normal'):
    voice_list = []
    for _ in range(num_samples):
        seq = []
        for _ in range(7):
            if scenario == 'normal':
                sr, ttr, rep, comp = np.random.uniform(3.5, 4.0), np.random.uniform(0.60, 0.75), np.random.uniform(0.02,
                                                                                                                   0.08), np.random.uniform(
                    18.0, 23.0)
            else:
                sr, ttr, rep, comp = np.random.uniform(2.0, 2.5), np.random.uniform(0.30, 0.40), np.random.uniform(0.20,
                                                                                                                   0.30), np.random.uniform(
                    8.0, 12.0)
            seq.append([sr, ttr, rep, comp])
        voice_list.append(seq)

    v_mean_np, v_std_np = v_mean.numpy(), v_std.numpy()
    if scenario == 'normal':
        noise = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, 30, 20))
        raw_vision = v_mean_np + (v_std_np * noise)
    else:
        noise = np.random.normal(loc=3.0, scale=0.1, size=(num_samples, 30, 20))
        raw_vision = v_mean_np + (v_std_np * noise)

    return torch.tensor(np.array(voice_list), dtype=torch.float32), torch.tensor(raw_vision, dtype=torch.float32)


# ==========================================
# 4. 앙상블 및 복합 임계값 캘리브레이션
# ==========================================
NUM_TESTS = 500
VOICE_WEIGHT = 0.5
VISION_WEIGHT = 0.5

# --- [정상 데이터 그룹 먼저 추론하여 앙상블 기준선 확보] ---
raw_voice_norm, raw_vision_norm = generate_ensemble_test_data(NUM_TESTS, vision_mean, vision_std, scenario='normal')
voice_mean = raw_voice_norm.mean(dim=(0, 1), keepdim=True)
voice_std = raw_voice_norm.std(dim=(0, 1), keepdim=True)

norm_voice_n = (raw_voice_norm - voice_mean) / (voice_std + 1e-7)
norm_vision_n = (raw_vision_norm - vision_mean) / (vision_std + 1e-7)

with torch.no_grad():
    v_loss_n = criterion(voice_model(norm_voice_n), norm_voice_n).mean(dim=(1, 2)).numpy()
    vis_loss_n = criterion(vision_model(norm_vision_n), norm_vision_n).mean(dim=(1, 2)).numpy()

# 정상 그룹의 최종 앙상블 위험 점수들 분포 계산
normal_ensemble_scores = (v_loss_n * VOICE_WEIGHT) + (vis_loss_n * VISION_WEIGHT)

# [방법 2] 최종 앙상블 위험 점수에 대한 복합 임계값(Ensemble Threshold) 선언
# 정상적인 상태에서 발생할 수 있는 종합 오차의 평균에 여유 버퍼를 더해 기준을 세우기.
ENSEMBLE_THRESHOLD = normal_ensemble_scores.mean() + (2 * normal_ensemble_scores.std())
print(f"[방법 2] 자동 교정된 최종 앙상블 종합 임계값: {ENSEMBLE_THRESHOLD:.6f}\n")

# ==========================================
# 5. 최종 보정 파이프라인 검증
# ==========================================
for stage in ['normal', 'abnormal']:
    if stage == 'normal':
        v_losses, vis_losses, ensemble_scores = v_loss_n, vis_loss_n, normal_ensemble_scores
    else:
        raw_voice_ab, raw_vision_ab = generate_ensemble_test_data(NUM_TESTS, vision_mean, vision_std,
                                                                  scenario='abnormal')
        norm_voice_a = (raw_voice_ab - voice_mean) / (voice_std + 1e-7)
        norm_vision_a = (raw_vision_ab - vision_mean) / (vision_std + 1e-7)
        with torch.no_grad():
            v_losses = criterion(voice_model(norm_voice_a), norm_voice_a).mean(dim=(1, 2)).numpy()
            vis_losses = criterion(vision_model(norm_vision_a), norm_vision_a).mean(dim=(1, 2)).numpy()
        ensemble_scores = (v_losses * VOICE_WEIGHT) + (vis_losses * VISION_WEIGHT)

    print(f"=== [결과 리포트: {stage.upper()}] ===")
    print(f"음성 평균 오차: {v_losses.mean():.4f}")
    print(f"비전 평균 오차: {vis_losses.mean():.4f}")
    print(f"최종 종합 위험 점수: {ensemble_scores.mean():.4f}")

    # 종합 임계값을 기준으로 최종 알림 대상자(이상치) 분류
    final_alarms = np.sum(ensemble_scores > ENSEMBLE_THRESHOLD)

    if stage == 'normal':
        print(f"[오탐율] 정상 유저인데 알림이 발송된 건수: {final_alarms}/{NUM_TESTS} ({final_alarms / NUM_TESTS * 100:.1f}%)")
    else:
        print(f"[정탐율] 위험 고령자를 정확히 찾아내 알림 발송한 건수: {final_alarms}/{NUM_TESTS} ({final_alarms / NUM_TESTS * 100:.1f}%)")
    print("-" * 50)