import torch
import torch.nn as nn


# ==========================================
# 1. 모델 아키텍처 정의 (가중치를 덮어씌울 뼈대)
# ==========================================
# 추론만 하더라도 모델의 구조(Class)는 알아야 파이토치가 가중치를 채워 넣을 수 있습니다.
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
# 2. 모델 로드 및 설정
# ==========================================
print("저장된 모델을 불러오는 중...")
# 뼈대 생성 (학습할 때와 동일하게 input_dim=4, hidden_dim=16으로 설정)
model = VoiceLSTMAutoencoder(input_dim=4, hidden_dim=16)

# 학습된 가중치(.pth) 불러와서 덮어씌우기
model.load_state_dict(torch.load('voice_lstm_model.pth'))

# 추론(평가) 모드로 전환 (Dropout이나 BatchNorm 등이 있다면 동작을 고정함)
model.eval()
criterion = nn.MSELoss()
print("모델 로드 완료!\n")

# ==========================================
# 3. 모델 변별력 테스트 (가상 데이터 주입)
# ==========================================
print("--- 🔍 모델 변별력(이상 탐지) 테스트 ---")

with torch.no_grad():  # 추론할 때는 기울기(Gradient) 계산을 꺼서 메모리를 절약합니다.

    # [시나리오 A] 정상적인 패턴의 데이터
    # Z-score 정규화가 되었다고 가정하고 평균(0) 근처의 값을 가짐
    normal_test_data = torch.randn(1, 7, 4) * 0.5

    pred_normal = model(normal_test_data)
    normal_score = criterion(pred_normal, normal_test_data).item()

    # [시나리오 B] 치매 초기 징후가 나타난 비정상 패턴의 데이터
    # 시나리오 A의 데이터에서 발화 속도는 확 줄이고(-2.0), 반복 표현은 확 늘립니다(+3.0)
    abnormal_test_data = normal_test_data.clone()
    abnormal_test_data[0, :, 0] -= 2.0  # Feature 1: 발화 속도 저하
    abnormal_test_data[0, :, 2] += 3.0  # Feature 3: 반복 표현 비율 급증

    pred_abnormal = model(abnormal_test_data)
    abnormal_score = criterion(pred_abnormal, abnormal_test_data).item()

# ==========================================
# 4. 결과 출력
# ==========================================
print(f"[정상 패턴]의 변화 점수: {normal_score:.6f} (낮을수록 평소와 비슷함)")
print(f"[이상 징후]의 변화 점수: {abnormal_score:.6f} (높을수록 평소와 다름!)")
print(f"결과: 비정상 데이터가 정상 대비 약 {abnormal_score / normal_score:.1f}배 더 높은 위험 점수를 기록했습니다.")