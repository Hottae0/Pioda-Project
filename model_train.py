import zipfile
import json
import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from konlpy.tag import Okt
from collections import Counter
from datetime import datetime




class VoiceLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=8, num_layers=1):
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

# 형태소 분석기 초기화
okt = Okt()

def extract_features_from_stt(stt_text, record_time):
    """텍스트와 녹음 시간에서 4가지 지표를 추출하는 함수"""
    # 1. 발화 속도
    pure_text = stt_text.replace(" ", "")
    speech_rate = len(pure_text) / record_time if record_time > 0 else 0

    # 2. 어휘 다양성 (TTR)
    pos_tags = okt.pos(stt_text)
    content_words = [word for word, pos in pos_tags if pos in ['Noun', 'Verb']]
    ttr = len(set(content_words)) / len(content_words) if content_words else 0

    # 3. 반복 표현 비율
    words = stt_text.split()
    bigrams = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]
    repeated = sum(1 for count in Counter(bigrams).values() if count > 1)
    repetition_rate = repeated / len(bigrams) if bigrams else 0

    # 4. 문장 복잡도 (형태소 개수)
    complexity = len(pos_tags)

    return [speech_rate, ttr, repetition_rate, complexity]


def process_aihub_zip_to_tensor(zip_path, extract_dir='./aihub_data', seq_length=7):
    """ZIP 압축 해제 -> JSON 파싱 -> 화자별 정렬 -> 3D 텐서 변환"""

    # 1. ZIP 파일 압축 해제
    print(f"'{zip_path}' 압축 해제 중...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    # 2. 모든 JSON 파일 찾기
    json_files = glob.glob(os.path.join(extract_dir, '**', '*.json'), recursive=True)
    print(f"총 {len(json_files)}개의 JSON 파일을 찾았습니다.")

    extracted_data = []

    # 3. JSON 파일 순회하며 데이터 및 지표 추출
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stt = data["발화정보"]["stt"]
            recrd_time = float(data["발화정보"]["recrdTime"])
            recorder_id = data["녹음자정보"]["recorderId"]
            recrd_dt = datetime.strptime(data["발화정보"]["recrdDt"], "%Y-%m-%d %H:%M:%S")

            # 피처 5개 뽑기
            features = extract_features_from_stt(stt, recrd_time)

            extracted_data.append([recorder_id, recrd_dt] + features)
        except Exception as e:
            pass  # 손상된 파일이나 형식이 다른 JSON은 가볍게 패스

    # DataFrame으로 변환
    columns = ['recorder_id', 'date', 'speech_rate', 'ttr', 'repetition_rate', 'complexity']
    df = pd.DataFrame(extracted_data, columns=columns)

    # 4. 화자별, 시간순으로 정렬 (LSTM 시계열)
    df = df.sort_values(by=['recorder_id', 'date'])

    # 5. seq_length만큼 잘라서 3D 텐서로 만들기
    tensor_list = []
    grouped = df.groupby('recorder_id')

    for _, group in grouped:
        features = group[['speech_rate', 'ttr', 'repetition_rate', 'complexity']].values

        # 데이터가 seq_length(예: 7)보다 많으면 7개씩 잘라서 묶음
        for i in range(0, len(features) - seq_length + 1, seq_length):
            sequence = features[i: i + seq_length]
            tensor_list.append(sequence)

    if not tensor_list:
        print("시퀀스를 만들기에 데이터가 충분하지 않음.")
        return None

    # (Batch Size, Seq Length, 5) 형태의 PyTorch Tensor로 최종 변환
    final_tensor = torch.tensor(np.array(tensor_list), dtype=torch.float32)
    print(f"변환 완료! 최종 Tensor Shape: {final_tensor.shape}")

    return final_tensor


zip_filepath = "training_voice_data.zip"
lstm_input_tensor = process_aihub_zip_to_tensor(zip_filepath, seq_length=7)

voice_data = lstm_input_tensor

# 1. 데이터 정규화 (Z-score Normalization)
# 피처별로 평균 0, 표준편차 1로 스케일을 맞추기
mean = voice_data.mean(dim=(0, 1), keepdim=True)
std = voice_data.std(dim=(0, 1), keepdim=True)
voice_data_normalized = (voice_data - mean) / (std + 1e-7)

# 2. 미니 배치 데이터로더 생성 (배치 사이즈 32)
dataset = TensorDataset(voice_data_normalized, voice_data_normalized)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. 모델 세팅
model = VoiceLSTMAutoencoder(input_dim=4, hidden_dim=16)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. 배치 학습 루프
epochs = 1000  # 학습 횟수를 1000번으로 증가
for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    # 32개씩 잘라서 학습
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        reconstructed = model(batch_x)
        loss = criterion(reconstructed, batch_y)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        # 미니 배치들의 평균 Loss 출력
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.6f}')

torch.save(model.state_dict(), 'voice_lstm_model.pth')
print("모델 가중치가 'voice_lstm_model.pth'로 성공적으로 저장")