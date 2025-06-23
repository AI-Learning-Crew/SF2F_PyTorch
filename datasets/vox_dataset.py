"""
==============================================================================
SF2F VoxCeleb Dataset: Speech-to-Face Paired Data Processing
==============================================================================

Paper: "Speech2Face: Learning a Face from a Voice"
URL: https://arxiv.org/abs/2006.05888

이 모듈은 SF2F(Speech Fusion to Face) 논문에서 사용하는 VoxCeleb 데이터셋의
핵심 데이터 처리 클래스를 구현합니다.

SF2F 논문의 데이터 처리 핵심 특징 (Core Data Processing Features):
=============================================================================

? 1. Multi-Modal Data Pairing (논문 Section 3의 핵심 입력)
   - Speech Features: 로그 멜 스펙트로그램 (40-dimensional log mel-spectrograms)
   - Face Images: 128x128 얼굴 이미지 (VGG Face Detection 크롭)
   - Identity Labels: 신원 보존 학습을 위한 사람별 고유 ID

? 2. Temporal Audio Segmentation (논문의 음성 처리 전략)
   - Variable-Length Audio: 다양한 길이의 음성 세그먼트 처리
   - Sliding Window: 일관된 길이의 학습 단위로 분할
   - Random Sampling: 과적합 방지를 위한 무작위 세그먼트 선택

? 3. Face-Audio Synchronization (논문의 핵심 매핑 전략)
   - Same Identity Guarantee: 같은 사람의 음성과 얼굴 매칭 보장
   - Multiple Views: 한 사람의 다양한 얼굴/음성 조합으로 robustness 향상
   - Data Augmentation: 수평 플립, 세그먼트 시프트 등

? 4. Quality Control & Error Handling (실제 구현의 견고성)
   - Corrupted Data Detection: 손상된 이미지/음성 파일 자동 감지
   - Fallback Mechanisms: 오류 발생 시 대체 데이터 제공
   - Empty Directory Filtering: 빈 디렉토리 자동 제외

? 5. Flexible Data Modes (논문의 다양한 실험 설정 지원)
   - Single Segment Mode: 기본 배치별 학습용
   - Multi-Segment Mode: Fuser 훈련 및 평가용
   - All Faces Mode: 동일 인물의 모든 얼굴 분석용

Technical Implementation:
=============================================================================
- Mel Spectrogram Processing: 40-dim log mel features with temporal dynamics
- Image Preprocessing: VGG face cropping, ImageNet normalization
- Batch Collation: Variable-length audio handling with padding/cropping
- Memory Efficiency: Lazy loading and efficient caching strategies

이 데이터셋 클래스는 SF2F 논문의 음성-얼굴 매핑 학습을 위한
모든 데이터 처리 요구사항을 충족합니다.
"""

# 기본 라이브러리들
import json  # JSON 형태의 데이터 분할 정보 처리
import os  # 파일 시스템 접근
import numpy as np  # 수치 연산 (멜 스펙트로그램, 이미지 배열 처리)
import random  # 무작위 샘플링 (데이터 증강)
import pickle  # 멜 스펙트로그램 직렬화 데이터 로드
import PIL  # 이미지 처리 라이브러리
from PIL import Image  # PIL 이미지 객체
import torch  # PyTorch 텐서 처리
from torch.utils.data import Dataset, DataLoader  # PyTorch 데이터셋 기본 클래스들
from torch.utils.data.dataloader import default_collate  # 기본 배치 결합 함수
import torchvision.transforms as T  # 이미지 변환 (리사이즈, 정규화 등)

# 커스텀 유틸리티 함수들 (이미지/멜 스펙트로그램 전후처리)
try:
    from .utils import imagenet_preprocess, imagenet_deprocess_batch, \
        fast_imagenet_deprocess_batch, fast_mel_deprocess_batch
except:
    from utils import imagenet_preprocess, imagenet_deprocess_batch, \
        fast_imagenet_deprocess_batch, fast_mel_deprocess_batch

# ===================================================================
# SF2F 논문: VoxCeleb 데이터셋 기본 경로
# ===================================================================
VOX_DIR = os.path.join('./data', 'VoxCeleb')

class VoxDataset(Dataset):
    """
    ===================================================================
    SF2F 논문: VoxCeleb 음성-얼굴 쌍 데이터셋 클래스
    ===================================================================
    
    ? 논문의 핵심 데이터 처리 시스템 구현
    
    VoxCeleb1과 VoxCeleb2를 통합하여 SF2F 훈련에 필요한 
    음성-얼굴 쌍 데이터를 효율적으로 제공합니다.
    
    핵심 기능:
    - Multi-modal data pairing: 동일 인물의 음성과 얼굴 매칭
    - Temporal segmentation: 가변 길이 음성의 일관된 처리
    - Quality control: 손상된 데이터 자동 감지 및 처리
    - Flexible modes: 다양한 훈련/평가 설정 지원
    """
    
    def __init__(self,
                 data_dir,
                 image_size=(64, 64),
                 face_type='masked',
                 image_normalize_method='imagenet',
                 mel_normalize_method='vox_mel',
                 nframe_range=(100, 150),
                 split_set='train',
                 split_json=os.path.join(VOX_DIR, 'split.json'),
                 return_mel_segments=False,
                 mel_seg_window_stride=(125, 125),
                 image_left_right_avg=False,
                 image_random_hflip=False):
        """
        ===================================================================
        SF2F VoxCeleb 데이터셋 초기화
        ===================================================================
        
        ? 논문 Section 4: Experimental Setup의 데이터 설정 구현
        
        Args:
            data_dir: VoxCeleb 데이터 루트 디렉토리
            image_size: 출력 이미지 크기 (SF2F에서는 (128, 128) 표준)
            face_type: 얼굴 타입
                - 'masked': 배경 마스크 처리된 얼굴 (SF2F 표준)
                - 'original': 원본 이미지
            image_normalize_method: 이미지 정규화 방법
                - 'imagenet': ImageNet 표준 정규화 (SF2F 사용)
                - 'standard': 표준 정규화
            mel_normalize_method: 멜 스펙트로그램 정규화
                - 'vox_mel': VoxCeleb 특화 정규화 (SF2F 표준)
            nframe_range: 멜 스펙트로그램 프레임 수 범위 (시간축 길이)
            split_set: 데이터 분할 ('train', 'val', 'test')
            split_json: 데이터 분할 정보 JSON 파일
            return_mel_segments: 멀티 세그먼트 모드 활성화
                - False: 단일 세그먼트 (일반 훈련)
                - True: 다중 세그먼트 (Fuser 훈련, 평가)
            mel_seg_window_stride: 멜 세그먼트 윈도우 (크기, 스트라이드)
            image_left_right_avg: 좌우 플립 평균화 (데이터 증강)
            image_random_hflip: 무작위 수평 플립 (훈련 시 증강)
            
        ? SF2F 논문의 데이터 처리 전략:
        - Multi-resolution support: 다양한 이미지 해상도 지원
        - Temporal dynamics: 음성의 시간적 변화 포착
        - Identity preservation: 사람별 일관된 매핑 학습
        """
        # ===================================================================
        # SF2F 논문: 기본 데이터 설정
        # ===================================================================
        self.data_dir = data_dir  # VoxCeleb 데이터 루트 경로
        self.image_size = image_size  # 출력 이미지 해상도 (SF2F: 128x128)
        self.face_type = face_type  # 얼굴 타입 (SF2F: 'masked' 선호)
        self.face_dir = self.face_type + '_faces'  # 얼굴 이미지 디렉토리명
        
        # ===================================================================
        # SF2F 논문: 정규화 및 전처리 설정
        # ===================================================================
        self.image_normalize_method = image_normalize_method  # 이미지 정규화
        self.mel_normalize_method = mel_normalize_method  # 멜 정규화
        
        # ===================================================================
        # SF2F 논문: 시간적 음성 처리 설정
        # ===================================================================
        self.nframe_range = nframe_range  # 멜 스펙트로그램 시간축 범위
        # SF2F에서 일관된 입력 길이를 위해 사용 (예: 100-150 프레임)
        
        # ===================================================================
        # SF2F 논문: 데이터 분할 및 모드 설정
        # ===================================================================
        self.split_set = split_set  # 현재 사용할 데이터 분할
        self.split_json = split_json  # 분할 정보 파일
        
        # ===================================================================
        # SF2F 논문: 고급 음성 처리 모드
        # ===================================================================
        self.return_mel_segments = return_mel_segments
        # ? False: 단일 멜 세그먼트 반환 (일반 GAN 훈련)
        # ? True: 다중 멜 세그먼트 반환 (Fuser 훈련, 평가)
        
        self.mel_seg_window_stride = mel_seg_window_stride
        # ? 슬라이딩 윈도우 설정 (윈도우 크기, 스트라이드)
        # SF2F에서 시간적 일관성 있는 세그먼트 생성에 사용
        
        self.shuffle_mel_segments = True
        # ? 멜 세그먼트 셔플링으로 시간적 과적합 방지
        
        self.mel_segments_rand_start = False
        # ? 무작위 시작점으로 데이터 다양성 증가
        # Fuser 데이터에서 특히 중요 (시간적 변화 포착)
        
        # ===================================================================
        # SF2F 논문: 이미지 데이터 증강 설정
        # ===================================================================
        self.image_left_right_avg = image_left_right_avg
        # ? 좌우 대칭 평균화: 얼굴의 대칭성 활용한 데이터 증강
        
        self.image_random_hflip = image_random_hflip
        # ? 무작위 수평 플립: 훈련 시 일반화 성능 향상

        # ===================================================================
        # SF2F 논문: 초기화 프로세스 실행
        # ===================================================================
        self.load_split_dict()  # 데이터 분할 정보 로드
        self.list_available_names()  # 사용 가능한 데이터 목록 구축
        self.set_image_transform()  # 이미지 변환 파이프라인 설정
        self.set_mel_transform()  # 멜 스펙트로그램 변환 파이프라인 설정

    def __len__(self):
        """데이터셋 크기 반환 (사용 가능한 사람 수)"""
        return len(self.available_names)

    def set_length(self, length):
        """데이터셋 크기 제한 (디버깅, 빠른 실험용)"""
        self.available_names = self.available_names[0:length]

    def __getitem__(self, index):
        """
        ===================================================================
        SF2F 논문: 메인 데이터 로딩 함수
        ===================================================================
        
        ? 논문의 핵심: 주어진 인덱스에서 음성-얼굴 쌍 반환
        
        SF2F 훈련의 기본 단위인 (이미지, 음성, 신원) 삼조를 생성합니다.
        동일한 사람의 음성과 얼굴을 무작위로 선택하여 매핑 학습을 지원합니다.
        
        Args:
            index: 사람 인덱스 (0 ~ len(dataset)-1)
            
        Returns:
            image: 얼굴 이미지 텐서 (C, H, W)
            log_mel: 로그 멜 스펙트로그램 텐서 (F, T) 또는 (N, F, T)
            human_id: 사람 ID 텐서 (신원 보존 학습용)
            
        ? SF2F의 핵심 데이터 무결성:
        - Same Identity: 반드시 동일 인물의 음성과 얼굴 매칭
        - Quality Assurance: 손상된 파일 자동 감지 및 처리
        - Random Sampling: 과적합 방지를 위한 무작위 선택
        """
        # ===================================================================
        # SF2F 논문: 신원 정보 추출
        # ===================================================================
        sub_dataset, name = self.available_names[index]
        # sub_dataset: 'vox1' 또는 'vox2'
        # name: 개별 사람의 고유 식별자 (예: 'id10001')
        
        # ===================================================================
        # SF2F 논문: 얼굴 이미지 로딩 (Quality Control 포함)
        # ===================================================================
        # ? 얼굴 이미지 디렉토리 경로 구성
        image_dir = os.path.join(
            self.data_dir, sub_dataset, self.face_dir, name)
        
        # ? 손상된 이미지 처리를 위한 Robust Loading
        image_files = os.listdir(image_dir)
        max_attempts = len(image_files)
        
        for attempt in range(max_attempts):
            try:
                # ? 무작위 얼굴 이미지 선택 (SF2F의 다양성 확보 전략)
                image_jpg = random.choice(image_files)
                image_path = os.path.join(image_dir, image_jpg)

                # ? 안전한 이미지 로딩 및 전처리
                with open(image_path, 'rb') as f:
                    with PIL.Image.open(f) as image:
                        WW, HH = image.size
                        image = image.convert('RGB')  # RGB 형식 보장
                        
                        # ? SF2F 데이터 증강: 좌우 대칭 평균화
                        if self.image_left_right_avg:
                            # 원본과 수평 플립의 평균으로 대칭성 강화
                            arr = (np.array(image) / 2.0 + \
                                np.array(T.functional.hflip(image)) / 2.0).astype(
                                    np.uint8)
                            image = PIL.Image.fromarray(arr, mode="RGB")
                            
                        # ? 이미지 변환 파이프라인 적용
                        # (리사이즈, 정규화, 텐서 변환 등)
                        image = self.image_transform(image)
                        break  # 성공적으로 로드되면 루프 종료
                        
            except (PIL.UnidentifiedImageError, OSError, IOError) as e:
                # ? 손상된 이미지 파일 처리
                print(f"WARNING: Skipping corrupted image {image_path}: {e}")
                # 손상된 파일을 목록에서 제거하여 재선택 방지
                if image_jpg in image_files:
                    image_files.remove(image_jpg)
                if not image_files:  # 모든 이미지가 손상된 경우
                    print(f"ERROR: All images corrupted for {name} in {sub_dataset}")
                    # ? Fallback: 기본 검은색 이미지 생성
                    image = torch.zeros((3, self.image_size[0], self.image_size[1]))
                    break
                continue
        else:
            # ? 모든 시도 실패 시 기본 이미지 생성
            print(f"ERROR: Could not load any image for {name} in {sub_dataset}")
            image = torch.zeros((3, self.image_size[0], self.image_size[1]))

        # ===================================================================
        # SF2F 논문: 멜 스펙트로그램 로딩 (Quality Control 포함)
        # ===================================================================
        # ? 멜 스펙트로그램 디렉토리 경로 구성
        mel_gram_dir = os.path.join(
            self.data_dir, sub_dataset, 'mel_spectrograms', name)
        
        # ? 디렉토리 존재 및 내용 확인 (오류 시에만 디버그 정보 출력)
        if os.path.exists(mel_gram_dir):
            files_in_dir = os.listdir(mel_gram_dir)
            if len(files_in_dir) == 0:
                print(f"ERROR: Empty directory found!")
                print(f"DEBUG: mel_gram_dir = {mel_gram_dir}")
                print(f"DEBUG: sub_dataset = {sub_dataset}, name = {name}")
                print(f"DEBUG: Files in mel_gram_dir: {files_in_dir}")
        else:
            print(f"ERROR: Directory does not exist!")
            print(f"DEBUG: mel_gram_dir = {mel_gram_dir}")
            print(f"DEBUG: sub_dataset = {sub_dataset}, name = {name}")

        # ? 손상된 멜 스펙트로그램 처리를 위한 Robust Loading
        mel_files = os.listdir(mel_gram_dir)
        max_mel_attempts = len(mel_files)
        
        for attempt in range(max_mel_attempts):
            try:
                # ? 무작위 멜 스펙트로그램 선택
                mel_gram_pickle = random.choice(mel_files)
                mel_gram_path = os.path.join(mel_gram_dir, mel_gram_pickle)
                
                if not self.return_mel_segments:
                    # ? 단일 세그먼트 모드 (일반 GAN 훈련)
                    log_mel = self.load_mel_gram(mel_gram_path)
                    log_mel = self.mel_transform(log_mel)
                    break
                    
            except (EOFError, pickle.UnpicklingError, OSError, IOError, KeyError) as e:
                # ? 손상된 멜 파일 처리
                print(f"WARNING: Skipping corrupted mel file {mel_gram_path}: {e}")
                if mel_gram_pickle in mel_files:
                    mel_files.remove(mel_gram_pickle)
                if not mel_files:
                    print(f"ERROR: All mel files corrupted for {name} in {sub_dataset}")
                    # ? Fallback: 기본 멜 스펙트로그램 생성
                    log_mel = torch.zeros((40, 100))  # 표준 크기 (40-dim mel, 100 frames)
                    break
                continue
        else:
            if self.return_mel_segments:
                # ? 다중 세그먼트 모드 (Fuser 훈련, 평가)
                # 이 모드에서는 한 사람의 모든 멜 세그먼트를 반환
                log_mel = self.get_all_mel_segments_of_id(
                    index, shuffle=self.shuffle_mel_segments)
            else:
                print(f"ERROR: Could not load any mel file for {name} in {sub_dataset}")
                log_mel = torch.zeros((40, 100))

        # ===================================================================
        # SF2F 논문: 신원 레이블 생성
        # ===================================================================
        # ? 신원 보존 학습을 위한 인덱스 기반 ID
        # 보조 분류기들이 이 ID로 신원 분류를 학습
        human_id = torch.tensor(index)

        return image, log_mel, human_id

    def get_all_faces_of_id(self, index):
        """
        ===================================================================
        SF2F 논문: 동일 인물의 모든 얼굴 이미지 로딩
        ===================================================================
        
        ? 논문의 평가 전략: 한 사람의 모든 얼굴로 평균 임베딩 계산
        
        이 함수는 SF2F의 고급 평가 모드에서 사용됩니다:
        - 여러 얼굴 이미지의 FaceNet 임베딩 평균화
        - 조명, 각도 변화에 robust한 대표 임베딩 생성
        - Recall@K 계산의 더 안정적인 기준 제공
        
        Args:
            index: 사람 인덱스
            
        Returns:
            faces: 모든 얼굴 이미지 배치 텐서 (N, C, H, W)
        """
        sub_dataset, name = self.available_names[index]
        faces = []
        
        # ? 해당 인물의 얼굴 이미지 디렉토리
        image_dir = os.path.join(
            self.data_dir, sub_dataset, self.face_dir, name)
            
        # ? 모든 얼굴 이미지를 순차적으로 로드
        for image_jpg in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_jpg)
            with open(image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
                    # 동일한 전처리 파이프라인 적용
                    image = self.image_transform(image.convert('RGB'))
                    faces.append(image)
                    
        # ? 배치 텐서로 결합 (N, C, H, W)
        faces = torch.stack(faces)
        return faces

    def get_all_mel_segments_of_id(self,
                                   index,
                                   shuffle=False):
        """
        ===================================================================
        SF2F 논문: 동일 인물의 모든 음성 세그먼트 로딩
        ===================================================================
        
        ? 논문의 핵심: 시간적 일관성 있는 멀티 세그먼트 처리
        
        이 함수는 SF2F의 Fuser 컴포넌트 훈련과 평가에서 사용됩니다:
        - 슬라이딩 윈도우로 일관된 길이의 세그먼트 생성
        - 시간적 변화를 포착하는 다양한 음성 특징 제공
        - 음성 임베딩 평균화를 통한 안정적인 얼굴 생성
        
        Args:
            index: 사람 인덱스
            shuffle: 멜 파일 순서 셔플링 여부
            
        Returns:
            segments: 모든 음성 세그먼트 배치 텐서 (N, F, T)
        """
        sub_dataset, name = self.available_names[index]
        window_length, stride_length = self.mel_seg_window_stride
        segments = []
        
        # ? 해당 인물의 멜 스펙트로그램 디렉토리
        mel_gram_dir = os.path.join(
            self.data_dir, sub_dataset, 'mel_spectrograms', name)
            
        # ? 멜 파일 목록 준비
        mel_gram_list = os.listdir(mel_gram_dir)
        if shuffle:
            random.shuffle(mel_gram_list)  # 시간적 순서 무작위화
        else:
            mel_gram_list.sort()  # 일관된 순서 유지
            
        seg_count = 0
        
        # ? 각 멜 스펙트로그램 파일에서 세그먼트 추출
        for mel_gram_pickle in mel_gram_list:
            mel_gram_path = os.path.join(mel_gram_dir, mel_gram_pickle)
            log_mel = self.load_mel_gram(mel_gram_path)
            log_mel = self.mel_transform(log_mel)
            mel_length = log_mel.shape[1]
            
            # ? 무작위 시작점 설정 (데이터 다양성 증가)
            if self.mel_segments_rand_start:
                start = np.random.randint(mel_length - window_length) if mel_length > window_length else 0
                log_mel = log_mel[:, start:]
                mel_length = log_mel.shape[1]
                
            # ? 슬라이딩 윈도우로 세그먼트 생성 가능 수 계산
            num_window = 1 + (mel_length - window_length) // stride_length
            
            # ? 슬라이딩 윈도우 적용
            for i in range(0, num_window):
                start_time = i * stride_length
                segment = log_mel[:, start_time:start_time + window_length]
                segments.append(segment)
                seg_count = seg_count + 1
                
                # ? 세그먼트 수 제한 (메모리 효율성)
                if seg_count == 20:  # 최대 20개 세그먼트
                    segments = torch.stack(segments)
                    return segments
                    
        segments = torch.stack(segments)
        return segments

    def set_image_transform(self):
        """
        ===================================================================
        SF2F 논문: 이미지 전처리 파이프라인 설정
        ===================================================================
        
        ? 논문의 표준 이미지 전처리 과정 구현
        
        SF2F에서 사용하는 이미지 전처리 단계:
        1. 무작위 수평 플립 (훈련 시 데이터 증강)
        2. 크기 조정 (목표 해상도로 리사이즈)
        3. 텐서 변환 (PIL → PyTorch Tensor)
        4. 정규화 (ImageNet 표준 또는 커스텀)
        """
        print('Dataloader: called set_image_size', self.image_size)
        
        # ? 기본 변환 파이프라인
        image_transform = [T.Resize(self.image_size), T.ToTensor()]
        
        # ? 훈련 시 데이터 증강 추가
        if self.image_random_hflip and self.split_set == 'train':
            # 50% 확률로 수평 플립 적용 (얼굴 대칭성 활용)
            image_transform = [T.RandomHorizontalFlip(p=0.5),] + \
                image_transform
                
        # ? 정규화 적용
        if self.image_normalize_method is not None:
            print('Dataloader: called image_normalize_method',
                self.image_normalize_method)
            # ImageNet 표준 정규화 또는 커스텀 정규화
            image_transform.append(imagenet_preprocess(
                normalize_method=self.image_normalize_method))
                
        self.image_transform = T.Compose(image_transform)

    def set_mel_transform(self):
        """
        ===================================================================
        SF2F 논문: 멜 스펙트로그램 전처리 파이프라인 설정
        ===================================================================
        
        ? 논문의 표준 음성 특징 전처리 과정 구현
        
        SF2F에서 사용하는 멜 스펙트로그램 전처리 단계:
        1. 텐서 변환 (NumPy → PyTorch Tensor)
        2. 정규화 (VoxCeleb 특화 정규화)
        3. 차원 압축 (squeeze)
        """
        # ? 기본 변환 파이프라인
        mel_transform = [T.ToTensor(), ]
        
        print('Dataloader: called mel_normalize_method',
            self.mel_normalize_method)
            
        # ? 멜 스펙트로그램 정규화 적용
        if self.mel_normalize_method is not None:
            # VoxCeleb 데이터셋에 특화된 정규화 방식
            mel_transform.append(imagenet_preprocess(
                normalize_method=self.mel_normalize_method))
                
        # ? 불필요한 차원 제거
        mel_transform.append(torch.squeeze)
        
        self.mel_transform = T.Compose(mel_transform)

    def load_split_dict(self):
        """
        ===================================================================
        SF2F 논문: 데이터 분할 정보 로딩
        ===================================================================
        
        ? 논문의 평가 프로토콜: 사람별 분할으로 일반화 성능 측정
        
        split.json에서 다음 정보를 로드:
        - train: 훈련용 사람 ID 리스트
        - val: 검증용 사람 ID 리스트  
        - test: 테스트용 사람 ID 리스트
        
        중요: SF2F에서는 사람 단위로 분할하여 identity leakage 방지
        """
        with open(self.split_json) as json_file:
            self.split_dict = json.load(json_file)

    def list_available_names(self):
        """
        ===================================================================
        SF2F 논문: 사용 가능한 데이터 목록 구축
        ===================================================================
        
        ? 논문의 데이터 무결성 보장: 음성과 얼굴 모두 존재하는 사람만 선택
        
        과정:
        1. VoxCeleb1과 VoxCeleb2에서 데이터 스캔
        2. 멜 스펙트로그램과 얼굴 이미지 교집합 계산
        3. 현재 분할(train/val/test)에 속하는 사람만 필터링
        4. 빈 디렉토리 제외 (품질 관리)
        """
        self.available_names = []
        
        # ? VoxCeleb1과 VoxCeleb2 통합 처리
        for sub_dataset in ('vox1', 'vox2'):
            # 멜 스펙트로그램 사용 가능한 사람 목록
            mel_gram_available = os.listdir(
                os.path.join(self.data_dir, sub_dataset, 'mel_spectrograms'))
            # 얼굴 이미지 사용 가능한 사람 목록
            face_available = os.listdir(
                os.path.join(self.data_dir, sub_dataset, self.face_dir))
                
            # ? 교집합 계산: 음성과 얼굴 모두 있는 사람만 선택
            available = \
                set(mel_gram_available).intersection(face_available)
                
            for name in available:
                # ? 현재 분할에 속하는지 확인
                if name in self.split_dict[sub_dataset][self.split_set]:
                    # ? 빈 디렉토리 확인 (품질 관리)
                    mel_dir = os.path.join(
                        self.data_dir, sub_dataset, 'mel_spectrograms', name)
                    face_dir = os.path.join(
                        self.data_dir, sub_dataset, self.face_dir, name)
                    
                    # ? 디렉토리 존재 및 내용 확인
                    if (os.path.exists(mel_dir) and len(os.listdir(mel_dir)) > 0 and
                        os.path.exists(face_dir) and len(os.listdir(face_dir)) > 0):
                        self.available_names.append((sub_dataset, name))
                    else:
                        print(f"WARNING: Skipping {name} in {sub_dataset} - empty directory")

        # ? 일관된 순서 보장
        self.available_names.sort()
        print(f"Total available samples after filtering: {len(self.available_names)}")

    def load_mel_gram(self, mel_pickle):
        """
        ===================================================================
        SF2F 논문: 멜 스펙트로그램 로딩 함수
        ===================================================================
        
        ? 논문의 음성 특징: 40차원 로그 멜 스펙트로그램
        
        Pickle 파일에서 다음 정보 로드:
        - LogMel_Features: 40-dimensional log mel spectrogram
        - spkid: 화자 ID
        - clipid: 클립 ID  
        - wavid: 음성 파일 ID
        
        Args:
            mel_pickle: 멜 스펙트로그램 pickle 파일 경로
            
        Returns:
            log_mel: 로그 멜 스펙트로그램 배열 (F, T)
        """
        try:
            # ? Pickle 파일에서 데이터 로드
            with open(mel_pickle, 'rb') as file:
                data = pickle.load(file)
            log_mel = data['LogMel_Features']
            return log_mel
        except (EOFError, pickle.UnpicklingError, OSError, IOError, KeyError) as e:
            print(f"ERROR: Failed to load mel file {mel_pickle}: {e}")
            # ? Fallback: 기본 멜 스펙트로그램 반환
            # 40차원은 일반적인 멜 특징 차원수
            return np.zeros((40, 100), dtype=np.float32)

    def crop_or_pad(self, log_mel, out_frame):
        """
        ===================================================================
        SF2F 논문: 멜 스펙트로그램 길이 정규화
        ===================================================================
        
        ? 논문의 배치 처리: 가변 길이 음성을 일정 길이로 통일
        
        collate_fn과 협력하여 배치 내 모든 멜 스펙트로그램을
        동일한 시간 길이로 맞춥니다.
        
        Args:
            log_mel: 입력 멜 스펙트로그램 텐서
            out_frame: 목표 프레임 수
            
        Returns:
            log_mel: 길이가 조정된 멜 스펙트로그램
        """
        freq, cur_frame = log_mel.shape
        
        if cur_frame >= out_frame:
            # ? 크롭: 무작위 위치에서 목표 길이만큼 자르기
            start = np.random.randint(0, cur_frame-out_frame+1)
            log_mel = log_mel[..., start:start+out_frame]
        else:
            # ? 패딩: 부족한 길이를 0으로 채우기
            zero_padding = np.zeros((freq, out_frame-cur_frame))
            zero_padding = self.mel_transform(zero_padding)
            if len(zero_padding.shape) == 1:
                zero_padding = zero_padding.view([-1, 1])
            log_mel = torch.cat([log_mel, zero_padding], -1)

        return log_mel

    def collate_fn(self, batch):
        """
        ===================================================================
        SF2F 논문: 커스텀 배치 결합 함수
        ===================================================================
        
        ? 논문의 배치 처리: 가변 길이 음성의 동적 길이 통일
        
        배치 내 모든 멜 스펙트로그램을 무작위로 선택된 
        동일한 길이로 통일합니다. 이는 다음을 보장합니다:
        - 효율적인 GPU 연산을 위한 동일 크기 텐서
        - 다양한 시간 길이 학습으로 일반화 성능 향상
        - 메모리 사용량 최적화
        
        Args:
            batch: [(image, log_mel, human_id), ...] 형태의 배치
            
        Returns:
            collated_batch: 통일된 길이의 배치
        """
        min_nframe, max_nframe = self.nframe_range
        assert min_nframe <= max_nframe
        
        # ? 배치별로 무작위 프레임 수 선택
        np.random.seed()
        num_frame = np.random.randint(min_nframe, max_nframe+1)

        # ? 각 샘플의 멜 스펙트로그램을 선택된 길이로 조정
        batch = [(item[0],
                  self.crop_or_pad(item[1], num_frame),
                  item[2]) for item in batch]
                  
        # ? 표준 PyTorch 배치 결합 적용
        return default_collate(batch)

    def count_faces(self):
        """
        ===================================================================
        데이터셋 통계: 전체 얼굴 이미지 수 계산
        ===================================================================
        """
        total_count = 0
        for index in range(len(self.available_names)):
            sub_dataset, name = self.available_names[index]
            # 얼굴 이미지 디렉토리
            image_dir = os.path.join(
                self.data_dir, sub_dataset, self.face_dir, name)
            cur_count = len(os.listdir(image_dir))
            total_count = total_count + cur_count
        print('Number of faces in current dataset: {}'.format(total_count))
        return total_count

    def count_speech(self):
        """
        ===================================================================
        데이터셋 통계: 전체 음성 파일 수 계산
        ===================================================================
        """
        total_count = 0
        for index in range(len(self.available_names)):
            sub_dataset, name = self.available_names[index]
            window_length, stride_length = self.mel_seg_window_stride
            # 멜 스펙트로그램 디렉토리
            mel_gram_dir = os.path.join(
                self.data_dir, sub_dataset, 'mel_spectrograms', name)
            mel_gram_list = os.listdir(mel_gram_dir)
            cur_count = len(mel_gram_list)
            total_count = total_count + cur_count
        print('Number of speech in current dataset: {}'.format(total_count))
        return total_count


if __name__ == '__main__':
    """
    ===================================================================
    SF2F VoxDataset 테스트 및 검증 코드
    ===================================================================
    
    이 섹션은 데이터셋 구현의 정확성을 검증하고
    다양한 사용 사례를 테스트합니다.
    """
    
    # ? 각 데이터 분할에 대한 기본 테스트
    for split_set in ['train', 'val', 'test']:
        vox_dataset = VoxDataset(
            data_dir=VOX_DIR,
            image_size=(64, 64),
            nframe_range=(300, 600),
            face_type='masked',
            image_normalize_method='imagenet',
            mel_normalize_method='vox_mel',
            split_set=split_set,
            split_json=os.path.join(VOX_DIR, 'split.json'))
        print('Length of {} set: {}'.format(split_set, len(vox_dataset)))
        vox_dataset.count_faces()
        vox_dataset.count_speech()

    # ? 배치 결합 함수 테스트
    loader_kwargs = {
        'batch_size': 16,
        'num_workers': 8,
        'shuffle': False,
        "drop_last": True,
        'collate_fn': vox_dataset.collate_fn,
    }
    val_loader = DataLoader(vox_dataset, **loader_kwargs)
    for iter, batch in enumerate(val_loader):
        images, log_mels, human_ids = batch
        print('log_mels.shape:', log_mels.shape)
        if iter > 10000:
            break
