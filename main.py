from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn

# 1. API 서버 및 데이터 스키마 초기화
app = FastAPI(
    title="PiODa 멀티모달 인지 저하 탐지 API",
    description="음성(Voice) 및 안면(Vision) 데이터를 앙상블하여 치매 초기 징후를 판별합니다.",
    version="1.0.0"
)


class InferenceRequest(BaseModel):
    user_id: str
    voice_features: list  # [7, 4] 형태의 2차원 리스트 (7일치, 4개 피처)
    vision_features: list  # [30, 20] 형태의 2차원 리스트 (30프레임, 20개 피처)


# 2. PyTorch 모델 아키텍처 정의
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


# 3. 전역 변수 (메모리 로딩용)
voice_model = None
vision_model = None
vision_mean = None
vision_std = None

# 시뮬레이션에서 도출한 최종 앙상블 임계값 (필요시 수정!!!!!!!!!!!)
ENSEMBLE_THRESHOLD = 5.0

# 4. 서버 시작 시 가중치 및 통계 로드 (Startup Event)
@app.on_event("startup")
async def startup_event():
    global voice_model, vision_model, vision_mean, vision_std
    print("모델 가중치 및 정규화 통계 로딩 중...")

    try:
        # 모델 인스턴스 생성
        voice_model = VoiceLSTMAutoencoder()
        vision_model = VisionLSTMAutoencoder()

        # 가중치 덮어씌우기
        voice_model.load_state_dict(torch.load('voice_lstm_model.pth', map_location='cpu'))
        vision_model.load_state_dict(torch.load('vision_autoencoder.pt', map_location='cpu'))

        # 평가 모드 전환
        voice_model.eval()
        vision_model.eval()

        # 비전 정규화 통계 로드
        vision_norm_stats = torch.load('norm_stats.pt', map_location='cpu')
        vision_mean = vision_norm_stats['mean']
        vision_std = vision_norm_stats['std']

        print("모든 모델 및 통계 로드 완료! 서버 트래픽을 받을 준비가 되었습니다.")
    except Exception as e:
        print(f"🚨 로드 실패 (파일 경로를 확인하세요): {e}")


# 5. 이상 탐지 추론 API 엔드포인트
@app.post("/predict")
async def predict_anomaly(request: InferenceRequest):
    try:
        # 1. JSON 리스트 데이터를 PyTorch 텐서로 변환 (배치 차원 1 추가)
        voice_tensor = torch.tensor([request.voice_features], dtype=torch.float32)
        vision_tensor = torch.tensor([request.vision_features], dtype=torch.float32)

        # 2. 데이터 정규화 (Normalization)
        # 음성: 현재는 들어온 데이터 자체의 통계로 정규화
        v_mean = voice_tensor.mean(dim=(0, 1), keepdim=True)
        v_std = voice_tensor.std(dim=(0, 1), keepdim=True)
        norm_voice = (voice_tensor - v_mean) / (v_std + 1e-7)

        # 비전: 저장해둔 전역 통계치 사용
        norm_vision = (vision_tensor - vision_mean) / (vision_std + 1e-7)

        # 3. 모델 추론 (오차 계산)
        criterion = nn.MSELoss(reduction='none')
        with torch.no_grad():
            pred_voice = voice_model(norm_voice)
            voice_loss = criterion(pred_voice, norm_voice).mean().item()

            pred_vision = vision_model(norm_vision)
            vision_loss = criterion(pred_vision, norm_vision).mean().item()

        # 4. 앙상블 (Late Fusion) 및 최종 판독
        VOICE_WEIGHT = 0.6
        VISION_WEIGHT = 0.4
        final_score = (voice_loss * VOICE_WEIGHT) + (vision_loss * VISION_WEIGHT)

        # 앙상블 점수가 임계값을 넘으면 위험(True)으로 판정
        is_alert = bool(final_score > ENSEMBLE_THRESHOLD)

        # 5. 프론트엔드로 반환할 결과 포맷팅
        return {
            "user_id": request.user_id,
            "status": "success",
            "scores": {
                "voice_loss": round(voice_loss, 4),
                "vision_loss": round(vision_loss, 4),
                "final_ensemble_score": round(final_score, 4)
            },
            "requires_alert": is_alert,
            "threshold_used": ENSEMBLE_THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))