import torch
import torch.nn as nn
import numpy as np


# ==========================================
# 1. 모델 아키텍처 정의
# ==========================================
class VoiceLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, num_layers=1):
        super(VoiceLSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        seq_length = x.size(1)
        hidden_last = hidden[-1].unsqueeze(1).repeat(1, seq_length, 1)
        decoded_seq, _ = self.decoder_lstm(hidden_last)
        out = self.output_layer(decoded_seq)
        return out


# ==========================================
# 2. 대량의 랜덤 시퀀스 생성 함수
# ==========================================
def generate_bulk_data(num_samples, seq_length=7, scenario='normal'):
    """지정된 범위 내에서 랜덤으로 7일치 시계열 데이터를 대량 생성"""
    data = []
    for _ in range(num_samples):
        seq = []
        for _ in range(seq_length):
            if scenario == 'normal':
                # 건강한 고령자 패턴 범위
                sr = np.random.uniform(3.0, 4.0)  # 발화 속도
                ttr = np.random.uniform(0.60, 0.75)  # 어휘 다양성
                rep = np.random.uniform(0.02, 0.08)  # 반복 표현 비율
                comp = np.random.uniform(18.0, 23.0)  # 문장 복잡도
            else:
                # 치매 초기 징후 패턴 범위
                sr = np.random.uniform(2.0, 2.5)
                ttr = np.random.uniform(0.30, 0.40)
                rep = np.random.uniform(0.20, 0.30)
                comp = np.random.uniform(8.0, 12.0)
            seq.append([sr, ttr, rep, comp])
        data.append(seq)

    return torch.tensor(data, dtype=torch.float32)


# ==========================================
# 3. 모델 로드 및 추론 세팅
# ==========================================
model = VoiceLSTMAutoencoder(input_dim=4, hidden_dim=16)
try:
    model.load_state_dict(torch.load('voice_lstm_model.pth'))
except FileNotFoundError:
    print("⚠️ 'voice_lstm_model.pth' 파일을 찾을 수 없습니다.")

model.eval()
criterion = nn.MSELoss(reduction='none')  # 개별 샘플의 Loss를 보기 위해 none 설정

# ==========================================
# 4. 검증 실행 (각 1,000명의 유저 데이터 시뮬레이션)
# ==========================================
num_test_samples = 1000

# 1,000개의 정상/비정상 시퀀스 생성 (Shape: [1000, 7, 4])
tensor_normal = generate_bulk_data(num_test_samples, scenario='normal')
tensor_abnormal = generate_bulk_data(num_test_samples, scenario='abnormal')

# 정상 데이터를 기준으로 스케일링 (Z-score)
baseline_mean = tensor_normal.mean(dim=(0, 1), keepdim=True)
baseline_std = tensor_normal.std(dim=(0, 1), keepdim=True)

norm_normal = (tensor_normal - baseline_mean) / (baseline_std + 1e-7)
norm_abnormal = (tensor_abnormal - baseline_mean) / (baseline_std + 1e-7)

with torch.no_grad():
    # 정상 데이터 1,000개 추론
    pred_normal = model(norm_normal)
    # 각 샘플별 변화 점수(MSE) 계산 후 평균 산출
    scores_normal = criterion(pred_normal, norm_normal).mean(dim=(1, 2)).numpy()

    # 비정상 데이터 1,000개 추론
    pred_abnormal = model(norm_abnormal)
    scores_abnormal = criterion(pred_abnormal, norm_abnormal).mean(dim=(1, 2)).numpy()

# ==========================================
# 5. 통계적 결과 출력
# ==========================================
print(f"--- 📊 {num_test_samples}개 대규모 가상 데이터 검증 결과 ---")
print(f"[정상 그룹] 평균 변화 점수: {scores_normal.mean():.6f} (표준편차: {scores_normal.std():.4f})")
print(f"[이상 그룹] 평균 변화 점수: {scores_abnormal.mean():.6f} (표준편차: {scores_abnormal.std():.4f})")
print(f"\n이상 그룹이 정상 그룹 대비 평균 {scores_abnormal.mean() / scores_normal.mean():.1f}배 높은 점수를 기록했습니다.")