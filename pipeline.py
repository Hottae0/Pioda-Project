import os
import json
import glob
import subprocess
import numpy as np
import torch
from konlpy.tag import Okt
from collections import Counter, defaultdict
from datetime import datetime

# KoNLPy(Okt)는 형태소 분석에 JVM(Java)이 필요하다.
# JAVA_HOME이 설정돼 있지 않으면 macOS의 java_home 유틸로 자동 탐지한다.
# (Java 8 이상 설치 필요. 자동 탐지 실패 시 JAVA_HOME을 직접 export 할 것)
if "JAVA_HOME" not in os.environ:
    try:
        os.environ["JAVA_HOME"] = subprocess.check_output(
            ["/usr/libexec/java_home"], text=True
        ).strip()
    except Exception:
        pass  # 시스템 기본 JVM 사용

okt = Okt()


def extract_features(stt_text, record_time):
    """STT 텍스트 + 녹음시간 → 음성 지표 4개"""
    # 1. 발화 속도 (글자수 / 초)
    pure_text = stt_text.replace(" ", "")
    speech_rate = len(pure_text) / record_time if record_time > 0 else 0.0

    # 2. 어휘 다양성 TTR (고유 내용어 / 전체 내용어)
    pos_tags = okt.pos(stt_text)
    content_words = [w for w, p in pos_tags if p in ['Noun', 'Verb']]
    ttr = len(set(content_words)) / len(content_words) if content_words else 0.0

    # 3. 반복 표현 비율 (중복 바이그램 / 전체 바이그램)
    words = stt_text.split()
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    repeated = sum(1 for c in Counter(bigrams).values() if c > 1)
    repetition_rate = repeated / len(bigrams) if bigrams else 0.0

    # 4. 문장 복잡도 (형태소 수)
    complexity = float(len(pos_tags))

    return [speech_rate, ttr, repetition_rate, complexity]


def load_records_from_dir(data_dir, max_files=None):
    """
    압축 해제된 JSON 폴더에서 피처 추출
    data_dir 예: "aihub_data/[라벨]1.AI챗봇"
    """
    json_paths = glob.glob(f"{glob.escape(data_dir)}/**/*.json", recursive=True)
    if max_files:
        json_paths = json_paths[:max_files]

    total = len(json_paths)
    print(f"[pipeline] {total}개 JSON 처리 시작...")

    records = []
    for i, path in enumerate(json_paths):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            stt       = data["발화정보"]["stt"].strip()
            recrd_time = float(data["발화정보"]["recrdTime"])
            recorder_id = data["녹음자정보"]["recorderId"]
            recrd_dt  = datetime.strptime(data["발화정보"]["recrdDt"], "%Y-%m-%d %H:%M:%S")

            # 너무 짧은 발화 제외 (노이즈)
            if recrd_time < 1.0 or len(stt) < 5:
                continue

            features = extract_features(stt, recrd_time)
            records.append((recorder_id, recrd_dt, features))

        except Exception:
            continue

        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{total} 완료...")

    print(f"[pipeline] 유효 발화 수: {len(records)}")
    return records


def build_sequences(records, seq_length=7):
    """화자별 시간순 정렬 → (N, seq_length, 4) 텐서"""
    speaker_data = defaultdict(list)
    for recorder_id, recrd_dt, features in records:
        speaker_data[recorder_id].append((recrd_dt, features))

    sequences = []
    for entries in speaker_data.values():
        entries.sort(key=lambda x: x[0])
        feat_array = np.array([f for _, f in entries], dtype=np.float32)

        for i in range(0, len(feat_array) - seq_length + 1, seq_length):
            sequences.append(feat_array[i:i + seq_length])

    if not sequences:
        print("[pipeline] 시퀀스 생성 실패: 데이터 부족")
        return None

    tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    print(f"[pipeline] 시퀀스 생성 완료: {tensor.shape}  (N, seq_len={seq_length}, features=4)")
    return tensor
