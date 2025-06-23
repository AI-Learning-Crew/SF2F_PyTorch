"""
==============================================================================
SF2F Dataset Builder: VoxCeleb Speech-to-Face Dataset Construction
==============================================================================

Paper: "Speech2Face: Learning a Face from a Voice"
URL: https://arxiv.org/abs/2006.05888

이 모듈은 SF2F(Speech Fusion to Face) 논문에서 사용하는 VoxCeleb 데이터셋의
구축과 데이터 로더 생성을 담당합니다.

SF2F 논문의 데이터셋 활용 전략 (Dataset Utilization Strategy):
=============================================================================

📍 1. VoxCeleb Dataset (논문 Section 4의 실험 데이터)
   - VoxCeleb1 + VoxCeleb2: 대규모 음성-얼굴 쌍 데이터셋
   - 6,000여 명의 셀러브리티 음성과 얼굴 이미지
   - 실제 인터뷰, 연설 등에서 추출한 자연스러운 음성-얼굴 매핑

📍 2. Multi-Modal Data Pairing (논문의 핵심 데이터 구조)
   - Speech Features: 로그 멜 스펙트로그램 (Log Mel-Spectrograms)
   - Face Images: 128x128 해상도 얼굴 이미지
   - Identity Labels: 신원 보존 학습을 위한 사람별 고유 ID

📍 3. Data Split Strategy (논문 Section 4.1의 평가 프로토콜)
   - Training Set: 모델 훈련용 음성-얼굴 쌍
   - Validation Set: 하이퍼파라미터 튜닝 및 모델 선택
   - Test Set: 최종 성능 평가 (Recall@K, VGGFace Score 등)

📍 4. Data Preprocessing Pipeline (논문의 표준화 전략)
   - Face Detection & Cropping: VGG Face Detection 표준 경계 상자
   - Audio Processing: 16kHz 샘플링, 멜 스펙트로그램 변환
   - Normalization: ImageNet 정규화, VoxCeleb 멜 정규화

Technical Implementation:
=============================================================================
- Face Types: 'masked', 'original' (마스크 처리된 배경 vs 원본)
- Image Normalization: 'imagenet' 표준 정규화
- Mel Normalization: 'vox_mel' VoxCeleb 멜 정규화
- Batch Processing: 효율적인 GPU 활용을 위한 배치 처리

이 데이터 구축 시스템은 SF2F 논문의 음성-얼굴 매핑 학습을 위한
표준화된 데이터 파이프라인을 제공합니다.
"""

# 기본 라이브러리들
import json  # JSON 형태의 데이터 분할 정보 처리
import os  # 운영체제 파일 시스템 인터페이스
import os.path as osp  # 경로 처리 유틸리티
import numpy as np  # 수치 연산 (데이터 통계 등)

# 프로젝트 커스텀 모듈들
from datasets import VoxDataset  # VoxCeleb 데이터셋 클래스
from torch.utils.data import DataLoader  # PyTorch 데이터 로더

# ===================================================================
# SF2F 논문: VoxCeleb 데이터셋 기본 경로 설정
# ===================================================================
# VoxCeleb 데이터가 저장된 루트 디렉토리
# SF2F에서 사용하는 표준 데이터 구조를 따름
VOX_DIR = os.path.join('./data', 'VoxCeleb')


def build_vox_dsets(data_opts, batch_size, image_size):
    """
    ===================================================================
    SF2F 논문: VoxCeleb 데이터셋 구축 함수
    ===================================================================
    
    📍 논문 Section 4: Experimental Setup에서 사용하는 데이터셋 구성
    
    VoxCeleb1과 VoxCeleb2를 결합한 대규모 음성-얼굴 쌍 데이터셋을 구축합니다.
    SF2F 논문에서 사용하는 표준 데이터 설정을 적용합니다.
    
    Args:
        data_opts: 데이터 관련 설정 옵션들
            - root_dir: VoxCeleb 데이터 루트 경로
            - face_type: 얼굴 타입 ('masked' vs 'original')
            - image_normalize_method: 이미지 정규화 방법
            - mel_normalize_method: 멜 스펙트로그램 정규화 방법
            - split_json: 훈련/검증/테스트 분할 정보 파일
        batch_size: 배치 크기 (GPU 메모리와 훈련 안정성 고려)
        image_size: 출력 이미지 해상도 (SF2F에서는 주로 128x128 사용)
    
    Returns:
        train_dset, val_dset, test_dset: 각각의 데이터셋 객체들
        
    📍 SF2F 논문의 데이터 활용 특징:
    - Large-scale dataset: 6,000+ identities
    - Natural speech-face pairs: 실제 인터뷰/연설 데이터
    - Identity-preserving splits: 사람별로 분할하여 일반화 성능 평가
    """
    # ===================================================================
    # SF2F 논문: 데이터셋 공통 설정 매개변수
    # ===================================================================
    dset_kwargs = {
        # 📍 데이터 루트 디렉토리 (VoxCeleb1 + VoxCeleb2)
        'data_dir': osp.join(data_opts['root_dir']),
        
        # 📍 이미지 해상도 (SF2F에서는 128x128 표준)
        # 논문에서 Progressive GAN 구조로 다해상도 지원하지만 최종 출력은 128x128
        'image_size': image_size,
        
        # 📍 얼굴 타입 설정 (기본값: 'masked')
        # 'masked': 배경이 마스크 처리된 얼굴 (얼굴 영역에만 집중)
        # 'original': 원본 이미지 (배경 포함)
        # SF2F에서는 얼굴 특징에 집중하기 위해 주로 'masked' 사용
        'face_type': data_opts.get('face_type', 'masked'),
        
        # 📍 이미지 정규화 방법 (기본값: 'imagenet')
        # ImageNet 표준 정규화로 사전훈련된 모델들과 호환성 보장
        # Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
        'image_normalize_method': \
            data_opts.get('image_normalize_method', 'imagenet'),
            
        # 📍 멜 스펙트로그램 정규화 방법 (기본값: 'vox_mel')
        # VoxCeleb 데이터셋에 특화된 정규화 방식
        # 음성 특징의 동적 범위를 일관되게 조정
        'mel_normalize_method': \
            data_opts.get('mel_normalize_method', 'vox_mel'),
            
        # 📍 초기 데이터 분할: 훈련 세트로 시작
        'split_set': 'train',
        
        # 📍 데이터 분할 정보 JSON 파일
        # 훈련/검증/테스트로 사람별 분할된 ID 리스트
        # SF2F에서는 사람 단위로 분할하여 identity leakage 방지
        'split_json': \
            data_opts.get('split_json', os.path.join(VOX_DIR, 'split.json'))
    }
    
    # ===================================================================
    # SF2F 논문: 훈련 데이터셋 구축
    # ===================================================================
    # 📍 훈련 세트: SF2F 모델 학습을 위한 대부분의 데이터
    # 음성-얼굴 매핑 패턴 학습, GAN 훈련, 신원 분류기 훈련 등에 사용
    train_dset = VoxDataset(**dset_kwargs)
    
    # 📍 에포크당 반복 횟수 계산 (훈련 진행도 모니터링용)
    # 전체 훈련 데이터를 배치 크기로 나눈 값
    # SF2F의 Progressive Training에서 학습률 스케줄링 등에 활용
    iter_per_epoch = len(train_dset) // batch_size
    print('There are %d iterations per epoch' % iter_per_epoch)

    # ===================================================================
    # SF2F 논문: 검증 데이터셋 구축
    # ===================================================================
    # 📍 검증 세트: 하이퍼파라미터 튜닝과 모델 선택용
    # - Recall@K 성능 모니터링
    # - 최적 체크포인트 선택
    # - 조기 종료 (Early Stopping) 결정
    # 훈련에 사용되지 않은 별도의 사람들로 구성
    dset_kwargs['split_set'] = 'val'
    val_dset = VoxDataset(**dset_kwargs)

    # ===================================================================
    # SF2F 논문: 테스트 데이터셋 구축
    # ===================================================================
    # 📍 테스트 세트: 최종 성능 평가용
    # - 논문 Table 2의 Recall@K 수치 계산
    # - VGGFace Score, Inception Score 등 품질 메트릭
    # - Inter-Human Similarity 분석
    # 훈련과 검증에 전혀 사용되지 않은 holdout 데이터
    dset_kwargs['split_set'] = 'test'
    test_dset = VoxDataset(**dset_kwargs)

    return train_dset, val_dset, test_dset


def build_dataset(opts):
    """
    ===================================================================
    SF2F 논문: 데이터셋 빌더 팩토리 함수
    ===================================================================
    
    📍 논문에서 사용하는 데이터셋 타입에 따른 적절한 빌더 선택
    
    현재는 VoxCeleb만 지원하지만, 확장 가능한 구조로 설계:
    - 추후 다른 음성-얼굴 데이터셋 추가 가능
    - 각 데이터셋별 특화된 전처리 파이프라인 적용
    
    Args:
        opts: 데이터셋 설정 옵션들
            - dataset: 데이터셋 타입 ('vox' for VoxCeleb)
            - data_opts: 데이터셋별 세부 설정
            - batch_size: 배치 크기
            - image_size: 이미지 해상도
    
    Returns:
        train_dset, val_dset, test_dset: 구축된 데이터셋들
        
    📍 SF2F 논문에서 VoxCeleb 선택 이유:
    - Large-scale: 6,000+ identities, 1M+ utterances
    - High-quality: Celebrity interviews with clear audio-visual correspondence
    - Diverse: Multiple languages, ages, genders, and speaking styles
    """
    if opts["dataset"] == "vox":
        # 📍 VoxCeleb 데이터셋 구축
        # SF2F 논문의 표준 실험 설정
        return build_vox_dsets(opts["data_opts"], opts["batch_size"],
                              opts["image_size"])
    else:
        # 📍 지원하지 않는 데이터셋 타입 에러
        # 추후 다른 음성-얼굴 데이터셋 추가 시 여기에 구현
        raise ValueError("Unrecognized dataset: {}".format(opts["dataset"]))


def build_loaders(opts):
    """
    ===================================================================
    SF2F 논문: PyTorch 데이터 로더 구축 함수
    ===================================================================
    
    📍 논문의 훈련 파이프라인에서 사용하는 효율적인 데이터 로딩 시스템
    
    SF2F 훈련의 특징적인 데이터 로딩 요구사항:
    1. Large Batch Processing: GAN 훈련의 안정성을 위한 큰 배치 크기
    2. Efficient GPU Utilization: 멀티 워커를 통한 병렬 데이터 로딩
    3. Custom Collation: 음성-얼굴 쌍의 특수한 배치화 처리
    4. Deterministic Validation: 재현 가능한 검증 결과를 위한 고정된 순서
    
    Args:
        opts: 데이터 로더 설정 옵션들
            - batch_size: 배치 크기 (SF2F에서는 보통 8-32)
            - workers: 데이터 로딩 워커 수 (CPU 코어 수에 맞춰 설정)
            - 기타 데이터셋 관련 설정들
    
    Returns:
        train_loader, val_loader, test_loader: 각각의 데이터 로더들
        
    📍 SF2F 논문의 데이터 로딩 최적화:
    - Multi-worker loading: I/O 병목 해결
    - Pin memory: GPU 전송 속도 향상
    - Custom collate function: 가변 길이 음성 데이터 처리
    """
    # ===================================================================
    # SF2F 논문: 데이터셋 구축
    # ===================================================================
    # 앞서 정의한 팩토리 함수로 훈련/검증/테스트 데이터셋 생성
    train_dset, val_dset, test_dset = build_dataset(opts)

    # ===================================================================
    # SF2F 논문: 훈련 데이터 로더 설정
    # ===================================================================
    loader_kwargs = {
        # 📍 배치 크기: GAN 훈련의 안정성과 성능에 중요한 요소
        # SF2F에서는 보통 8-32 사용 (GPU 메모리에 따라 조정)
        # 너무 작으면 BatchNorm이 불안정, 너무 크면 메모리 부족
        'batch_size': opts["batch_size"],
        
        # 📍 워커 수: 병렬 데이터 로딩으로 I/O 병목 해결
        # CPU 코어 수의 2-4배 정도가 일반적
        # SF2F의 복잡한 전처리 파이프라인에서 특히 중요
        'num_workers': opts["workers"],
        
        # 📍 셔플링: 훈련 데이터의 무작위 순서로 모델 일반화 향상
        # SF2F에서 다양한 음성-얼굴 조합 학습을 위해 필수
        'shuffle': True,
        
        # 📍 마지막 배치 제거: 불완전한 배치로 인한 BatchNorm 오류 방지
        # GAN 훈련에서 배치 크기 일관성이 중요
        "drop_last": True,
        
        # 📍 커스텀 배치화 함수: 음성-얼굴 쌍의 특수 처리
        # - 가변 길이 음성 데이터의 패딩/크롭
        # - 이미지-음성 동기화 보장
        # - 신원 레이블 매칭
        'collate_fn': train_dset.collate_fn,
    }
    
    # 📍 훈련 데이터 로더 생성
    # SF2F 모델 훈련의 핵심 데이터 공급원
    # - 음성 특징 (로그 멜 스펙트로그램)
    # - 얼굴 이미지 (Ground Truth)
    # - 신원 레이블 (Identity Preservation 학습용)
    train_loader = DataLoader(train_dset, **loader_kwargs)
    
    # ===================================================================
    # SF2F 논문: 검증/테스트 데이터 로더 설정
    # ===================================================================
    # 📍 검증과 테스트에서는 셔플링과 드롭 라스트 비활성화
    # 재현 가능하고 일관된 평가를 위해 고정된 순서 사용
    loader_kwargs['shuffle'] = False
    loader_kwargs['drop_last'] = False
    
    # 📍 검증 데이터 로더 생성
    # - 에포크마다 Recall@K, VGGFace Score 등 계산
    # - 모델 선택과 하이퍼파라미터 튜닝에 사용
    # - 훈련 중 과적합 모니터링
    val_loader = DataLoader(val_dset, **loader_kwargs)
    
    # 📍 테스트 데이터 로더 생성
    # - 최종 논문 결과 도출용
    # - 논문 Table 2의 정량적 성능 수치 계산
    # - 검증에 사용되지 않은 완전한 holdout 데이터
    test_loader = DataLoader(test_dset, **loader_kwargs)
    
    return train_loader, val_loader, test_loader
