'''
==============================================================================
SF2F Evaluator: Speech-to-Face Performance Evaluation Module
==============================================================================

Paper: "Speech2Face: Learning a Face from a Voice" 
URL: https://arxiv.org/abs/2006.05888

이 파일은 SF2F(Speech Fusion to Face) 모델의 성능을 평가하기 위한 전용 평가기를 구현합니다.
training_utils.py, test.py와 협력하여 종합적인 평가 시스템을 구성합니다.

SF2F 논문의 핵심 평가 메트릭들 (Core Evaluation Metrics):
=============================================================================

📍 1. Identity Recall@K (논문 Table 2의 핵심 지표)
   - Recall@1, @5, @10: 음성에서 올바른 얼굴을 찾는 정확도
   - SF2F의 가장 중요한 성능 지표로, 음성-얼굴 매핑의 정확성을 직접 측정

📍 2. VGGFace Feature Space Similarity (논문 Section 4.1)
   - L1/L2 Distance: 얼굴 특징 공간에서의 거리 측정
   - Cosine Similarity: 특징 벡터 간의 방향성 유사도
   - 픽셀 단위가 아닌 고수준 얼굴 특징에서의 평가

📍 3. Inter-Human Similarity Analysis (논문의 혁신적 평가 접근)
   - 생성된 얼굴들 간의 다양성과 현실성 평가
   - 과도한 유사성(mode collapse) 방지 확인

📍 4. Multi-Modal Face Generation (논문 Section 3의 구현 검증)
   - 다양한 음성 세그먼트에서 일관된 얼굴 생성 능력 평가
   - 음성의 시간적 변화에 대한 얼굴 생성의 안정성 측정

Technical Implementation:
=============================================================================
- FaceNet (VGGFace2 pretrained): 얼굴 특징 추출을 위한 사전훈련된 모델
- VGG Face Cropping: 표준화된 얼굴 영역 추출 [0.235, 0.195, 0.765, 0.915]
- Multiple Query Methods: L1, L2, Cosine Similarity 기반 검색
- Batch Processing: 대규모 데이터셋 효율적 처리

Evaluation Modes:
=============================================================================
1. naive: 기본적인 배치별 평가
2. average_facenet_embedding: 여러 얼굴 이미지의 FaceNet 임베딩 평균화
3. average_voice_embedding: 음성 임베딩 평균화를 통한 안정적 얼굴 생성

이 평가기는 SF2F 논문에서 제시한 모든 정량적 평가 지표를 구현하여
음성-얼굴 매핑 모델의 성능을 종합적으로 평가합니다.
'''


import os
import json
import math
from collections import defaultdict
import time
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy
from torchvision.models import inception_v3
from pyprind import prog_bar
from tensorflow import gfile
from imageio import imwrite

from datasets import fast_imagenet_deprocess_batch, imagenet_deprocess_batch
import datasets
import models
from models import InceptionResnetV1, fixed_image_standardization


# ===================================================================
# SF2F 논문 Figure 3: VGG Face Detection의 표준 경계 상자
# 논문에서 얼굴 영역 표준화를 위해 사용하는 기준 좌표
# ===================================================================
# left, top, right, bottom - 얼굴 영역의 상대적 좌표
# SF2F에서 일관된 얼굴 비교를 위해 모든 이미지를 이 영역으로 크롭
VGG_BOX = [0.235, 0.195, 0.765, 0.915]


class S2fEvaluator:
    """
    ===================================================================
    SF2F 전용 평가기 클래스 (Speech-to-Face Evaluator)
    ===================================================================
    
    SF2F 논문의 핵심 평가 메트릭들을 구현하는 종합 평가 시스템:
    
    📍 논문 Table 2: Identity Recall@K 메트릭
    📍 논문 Section 4.1: VGGFace 특징 공간 유사도 분석
    📍 논문의 혁신적 평가: Inter-Human Similarity 측정
    📍 논문 Section 3: 다중 모달 얼굴 생성 평가
    
    이 클래스는 음성-얼굴 매핑의 정확성과 품질을 종합적으로 평가합니다.
    """
    
    def __init__(self,
                 loader,
                 options,
                 nframe_range=None,
                 extraction_size=100,
                 hq_emb_dict=True,
                 face_gen_mode='naive',
                 facenet_return_pooling=False):
        '''
        ===================================================================
        SF2F 평가기 초기화
        ===================================================================
        
        Args:
            loader: 데이터 로더 (VoxCeleb 음성-얼굴 쌍 데이터)
            options: 설정 옵션들 (이미지 정규화, 크롭 설정 등)
            extraction_size: 평가에 사용할 데이터 크기 (논문에서 100, 200, 300 사용)
            hq_emb_dict: 고품질 임베딩 딕셔너리 사용 여부
            face_gen_mode: 얼굴 생성 모드
                - 'naive': 기본 배치별 생성
                - 'average_facenet_embedding': FaceNet 임베딩 평균화
                - 'average_voice_embedding': 음성 임베딩 평균화
            facenet_return_pooling: FaceNet 풀링 레이어 반환 여부
        
        📍 논문 핵심: VGGFace2로 사전훈련된 FaceNet을 특징 추출기로 사용
        이는 SF2F 논문에서 얼굴 품질 평가의 기준으로 활용
        '''
        self.loader = deepcopy(loader)
        # Fuser 평가를 위해 동일한 멜 스펙트로그램 그룹 사용 보장
        # SF2F에서 일관된 음성 특징 사용을 위한 설정
        self.loader.dataset.shuffle_mel_segments = False
        if nframe_range is not None:
            self.loader.dataset.nframe_range = nframe_range
            
        # 📍 논문 핵심: VGGFace2 사전훈련된 FaceNet 사용
        # SF2F 논문 Section 4.1에서 얼굴 특징 추출의 표준으로 채택
        # 이는 얼굴 인식에서 state-of-the-art 성능을 보이는 모델
        self.facenet = InceptionResnetV1(
            pretrained='vggface2',
            auto_input_resize=True,
            return_pooling=facenet_return_pooling).cuda().eval()
        self.float_dtype = torch.cuda.FloatTensor
        self.long_dtype = torch.cuda.LongTensor
        self.options = options
        
        # 📍 논문에서 사용하는 이미지 정규화 방법
        # SF2F에서 일관된 이미지 처리를 위한 표준화
        self.image_normalize_method= \
                self.options["data"]["data_opts"]["image_normalize_method"]
        
        # 📍 논문 Figure 3: FaceNet 입력을 위한 전처리 파이프라인
        # ImageNet 정규화 해제 → FaceNet 표준화 적용
        self.do_deprocess_and_preprocess = \
            self.options["eval"]["facenet"]["deprocess_and_preprocess"]
            
        # 📍 SF2F 논문의 핵심: VGG 평균 경계 상자로 얼굴 크롭
        # 표준화된 얼굴 영역에서만 비교하여 정확한 평가 보장
        self.crop_faces = \
            self.options["eval"]["facenet"]["crop_faces"]
        self.extraction_size = extraction_size
        self.hq_emb_dict = hq_emb_dict
        self.face_gen_mode = face_gen_mode

        # 📍 논문 핵심: 데이터셋의 모든 얼굴에 대한 FaceNet 임베딩 사전 계산
        # 이는 Recall@K 계산을 위한 기준 임베딩 데이터베이스 구축
        self.get_dataset_embeddings()

        # ===================================================================
        # SF2F 논문 Section 4.1: 다양한 거리 메트릭 구현
        # ===================================================================
        # 📍 논문에서 사용하는 3가지 유사도 측정 방법
        # L2 Distance: 유클리드 거리 (일반적인 특징 공간 거리)
        self.L2_dist = \
            nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
        # L1 Distance: 맨하탄 거리 (robust한 거리 측정)
        self.L1_dist = \
            nn.PairwiseDistance(p=1.0, eps=1e-06, keepdim=False)
        # Cosine Similarity: 벡터 간의 각도 유사도 (방향성 고려)
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-08)

    def deprocess_and_preprocess(self, imgs):
        '''
        ===================================================================
        SF2F 논문 Figure 3: FaceNet 입력을 위한 이미지 전처리 파이프라인
        ===================================================================
        
        실제/생성된 이미지 배치에 대해 다음 과정을 수행:
        1. ImageNet 정규화 해제 (원본 이미지 복원)
        2. FaceNet 전용 표준화 적용
        
        📍 논문 핵심: 일관된 얼굴 특징 추출을 위한 표준화된 전처리
        SF2F에서 모든 얼굴 이미지는 동일한 전처리 과정을 거쳐야 함
        '''
        #print('Begin:', imgs[0])  # 디버깅용 출력
        # ImageNet 정규화 해제: 훈련 중 사용된 정규화를 되돌림
        imgs = fast_imagenet_deprocess_batch(
            imgs,
            normalize_method=self.image_normalize_method)
        #print('Our distribution:', imgs[0])  # 디버깅용 출력
        
        # 📍 FaceNet 전용 표준화: VGGFace2 사전훈련에 맞는 정규화
        # 이는 정확한 얼굴 특징 추출을 위해 필수적
        imgs = fixed_image_standardization(imgs)
        #print('fixed_image_standardization:', imgs[0])  # 디버깅용 출력
        return imgs

    def crop_vgg_box(self, imgs):
        '''
        ===================================================================
        SF2F 논문: VGG 평균 경계 상자를 사용한 표준화된 얼굴 크롭
        ===================================================================
        
        📍 논문 핵심: 모든 얼굴을 동일한 기준으로 크롭하여 공정한 비교
        VGG_BOX = [0.235, 0.195, 0.765, 0.915] (left, top, right, bottom)
        
        이는 얼굴 인식 분야의 표준 크롭 영역으로, SF2F에서도 채택
        배경이나 머리카락 등의 영향을 최소화하고 얼굴 영역에만 집중
        '''
        # VGG 표준 경계 상자 좌표 (상대적 비율)
        left, top, right, bottom = VGG_BOX
        # = [0.235015, 0.19505739, 0.76817876, 0.9154963]
        N, C, H, W = imgs.shape
        
        # 상대 좌표를 절대 픽셀 좌표로 변환
        left = int(left * W)
        right = int(right * W)
        top = int(top * H)
        bottom = int(bottom * H)
        
        # 얼굴 영역만 크롭 (표준화된 얼굴 비교를 위해)
        imgs = imgs[:, :, top:bottom+1, left:right+1]
        return imgs

    def get_dataset_embeddings(self):
        '''
        ===================================================================
        SF2F 논문: 기준 데이터셋의 FaceNet 임베딩 데이터베이스 구축
        ===================================================================
        
        📍 논문 핵심: Recall@K 계산을 위한 Ground Truth 임베딩 생성
        
        모든 실제 얼굴 이미지에 대해:
        1. VGGFace2 사전훈련된 FaceNet으로 특징 추출
        2. 표준화된 전처리 및 크롭 적용
        3. 고품질 임베딩 딕셔너리 구축 (hq_emb_dict=True일 때)
        
        이 임베딩들은 음성에서 생성된 얼굴과 비교하여
        "올바른 사람을 찾았는가?"를 판단하는 기준이 됩니다.
        '''
        with torch.no_grad():
            # BatchNorm 레이어의 running_mean/var에 영향을 주지 않도록 방지
            embedding_batches = []
            if self.hq_emb_dict:
                # 📍 고품질 임베딩 모드: 각 ID별 모든 얼굴의 평균 임베딩 사용
                # SF2F 논문에서 더 안정적인 평가를 위해 권장하는 방식
                for i in prog_bar(
                    range(len(self.loader.dataset)),
                    title="[S2fEvaluator: " + \
                        "Preparing FaceNet Embedding Dictionary]",
                    width=50):
                    # 각 ID의 모든 얼굴 이미지 로드
                    ######### unpack the data #########
                    imgs = self.loader.dataset.get_all_faces_of_id(i)
                    imgs = imgs.cuda()
                    ###################################
                    
                    # 📍 SF2F 표준 전처리 파이프라인 적용
                    if self.do_deprocess_and_preprocess:
                        imgs = self.deprocess_and_preprocess(imgs)
                    if self.crop_faces:
                        imgs = self.crop_vgg_box(imgs)
                        
                    # FaceNet으로 얼굴 특징 추출
                    embeddings = self.facenet(imgs)
                    # 📍 논문 핵심: 여러 얼굴 이미지의 평균 임베딩으로 안정성 향상
                    # 조명, 각도 등의 변화에 robust한 대표 임베딩 생성
                    embeddings = torch.mean(embeddings, 0, keepdim=True)
                    embedding_batches.append(embeddings)
            else:
                # 📍 기본 모드: 배치별 단일 임베딩 사용
                for batch in prog_bar(
                    self.loader,
                    title="[S2fEvaluator: " + \
                        "Preparing FaceNet Embedding Dictionary]",
                    width=50):
                    # Loop logic
                    ######### unpack the data #########
                    imgs, log_mels, human_ids = batch
                    imgs = imgs.cuda()
                    ###################################
                    if self.do_deprocess_and_preprocess:
                        imgs = self.deprocess_and_preprocess(imgs)
                    if self.crop_faces:
                        imgs = self.crop_vgg_box(imgs)
                    embeddings = self.facenet(imgs)
                    embedding_batches.append(embeddings)
                    
            # 📍 모든 임베딩을 하나의 텐서로 결합 (검색 데이터베이스 구성)
            self.dataset_embedding = torch.cat(embedding_batches, 0)
        print("S2fEvaluator: dataset_embedding shape:",
              self.dataset_embedding.shape)

    def get_pred_img_embeddings(self, model):
        '''
        ===================================================================
        SF2F 논문: 생성된 얼굴 이미지의 FaceNet 임베딩 추출
        ===================================================================
        
        📍 논문 핵심: 음성에서 생성된 얼굴들의 특징 벡터 계산
        
        이 함수는 SF2F 모델이 음성에서 생성한 얼굴들에 대해:
        1. 동일한 FaceNet으로 특징 추출
        2. 실제 얼굴과 동일한 전처리 적용
        3. 3가지 생성 모드 지원:
           - naive: 기본 배치별 생성
           - average_facenet_embedding: 생성된 여러 얼굴의 평균
           - average_voice_embedding: 음성 임베딩 평균화 활용
        
        이 임베딩들은 dataset_embedding과 비교되어 Recall@K를 계산합니다.
        '''
        training_status = model.training
        model.eval()  # 평가 모드로 전환 (드롭아웃 비활성화)
        pred_img_embedding_batches = []
        
        with torch.no_grad():
            # BatchNorm 레이어의 running_mean/var에 영향을 주지 않도록 방지
            if self.face_gen_mode == 'naive':
                # 📍 기본 모드: 배치별 얼굴 생성 및 평가
                for batch in prog_bar(
                    self.loader,
                    title="[S2fEvaluator: " + \
                        "Getting FaceNet Embedding for Predicted Images]",
                    width=50):
                    ######### unpack the data #########
                    imgs, log_mels, human_ids = batch
                    imgs = imgs.cuda()
                    log_mels = log_mels.type(self.float_dtype)
                    human_ids = human_ids.type(self.long_dtype)
                    ###################################
                    
                    # 📍 SF2F 모델 실행: 음성 → 얼굴 생성
                    # 훈련 중과 동일한 방식으로 모델 실행
                    model_out = model(log_mels)
                    imgs_pred, others = model_out
                    
                    # Multi-Resolution 출력 처리 (가장 높은 해상도 사용)
                    if isinstance(imgs_pred, tuple):
                        imgs_pred = imgs_pred[-1]
                        
                    # 📍 실제 얼굴과 동일한 전처리 적용 (공정한 비교를 위해)
                    if self.do_deprocess_and_preprocess:
                        imgs_pred = self.deprocess_and_preprocess(imgs_pred)
                    if self.crop_faces:
                        imgs_pred = self.crop_vgg_box(imgs_pred)
                        
                    # FaceNet으로 생성된 얼굴의 특징 추출
                    pred_img_embeddings = self.facenet(imgs_pred)
                    pred_img_embedding_batches.append(pred_img_embeddings)
                    
            elif self.face_gen_mode == 'average_facenet_embedding':
                # 📍 고급 모드: 여러 생성 얼굴의 FaceNet 임베딩 평균화
                # SF2F에서 더 안정적인 성능 평가를 위해 사용
                for i in prog_bar(
                    range(len(self.loader.dataset)),
                    title="[S2fEvaluator: " + \
                        "Getting FaceNet Embedding for Predicted Images, " + \
                            "with average Facenet embedding policy]",
                    width=50):
                    ######### unpack the data #########
                    # 각 ID의 모든 음성 세그먼트 로드
                    log_mels = self.loader.dataset.get_all_mel_segments_of_id(i)
                    log_mels = log_mels.type(self.float_dtype)
                    ###################################
                    
                    # 여러 음성 세그먼트에서 얼굴 생성
                    model_out = model(log_mels)
                    imgs_pred, others = model_out
                    if isinstance(imgs_pred, tuple):
                        imgs_pred = imgs_pred[-1]
                        
                    if self.do_deprocess_and_preprocess:
                        imgs_pred = self.deprocess_and_preprocess(imgs_pred)
                    if self.crop_faces:
                        imgs_pred = self.crop_vgg_box(imgs_pred)
                        
                    pred_img_embeddings = self.facenet(imgs_pred)
                    # 📍 핵심: 여러 생성 얼굴의 평균 임베딩으로 안정성 향상
                    # 음성의 시간적 변화에 robust한 대표 임베딩 생성
                    pred_img_embeddings = torch.mean(
                        pred_img_embeddings, 0, keepdim=True)
                    pred_img_embedding_batches.append(pred_img_embeddings)
                    
            elif self.face_gen_mode == 'average_voice_embedding':
                # 📍 최고급 모드: 음성 임베딩 평균화를 통한 안정적 얼굴 생성
                # SF2F에서 제안하는 가장 정교한 평가 방식
                for i in prog_bar(
                    range(len(self.loader.dataset)),
                    title="[S2fEvaluator: " + \
                        "Getting FaceNet Embedding for Predicted Images, " + \
                            "with average voice embedding policy]",
                    width=50):
                    ######### unpack the data #########
                    log_mels = self.loader.dataset.get_all_mel_segments_of_id(i)
                    log_mels = log_mels.type(self.float_dtype)
                    ###################################
                    
                    # 📍 SF2F 혁신: 음성 임베딩 평균화로 더 안정적인 얼굴 생성
                    model_out = model(log_mels, average_voice_embedding=True)
                    imgs_pred, others = model_out
                    if isinstance(imgs_pred, tuple):
                        imgs_pred = imgs_pred[-1]
                        
                    if self.do_deprocess_and_preprocess:
                        imgs_pred = self.deprocess_and_preprocess(imgs_pred)
                    if self.crop_faces:
                        imgs_pred = self.crop_vgg_box(imgs_pred)
                        
                    pred_img_embeddings = self.facenet(imgs_pred)
                    pred_img_embedding_batches.append(pred_img_embeddings)
                    
            # 모든 생성된 얼굴의 임베딩을 결합
            pred_img_embedding = torch.cat(pred_img_embedding_batches, 0)
            
        model.train(mode=training_status)  # 원래 훈련 상태로 복원
        return pred_img_embedding

    def L1_query(self, x, y):
        '''
        Given x, extract top K similar features from y, based on L1 distance
        x in shape (N_x, D)
        y in shape (N_y, D)
        '''
        # (N_x, D) --> (N_x, 1, D)
        x = x.unsqueeze(1)
        # Initialize: (N_x, )
        x_ids = torch.tensor(np.arange(x.shape[0])).cpu()
        # (N_y, D) --> (1, N_y, D)
        y = y.unsqueeze(0)
        # Output: (N_x, N_y, D)
        L1_table = torch.abs(x - y)
        # (N_x, N_y, D) --> (N_x, N_y)
        L1_table = torch.mean(L1_table, dim=-1)
        L1_table = torch.neg(L1_table)
        # Top K: (N_x, K)
        top_1_vals, top_1_indices = torch.topk(L1_table, 1, dim=-1)
        top_5_vals, top_5_indices = torch.topk(L1_table, 5, dim=-1)
        top_10_vals, top_10_indices = torch.topk(L1_table, 10, dim=-1)
        top_50_vals, top_50_indices = torch.topk(L1_table, 50, dim=-1)

        recall_at_1 = self.in_top_k(top_1_indices.cpu(), x_ids)
        recall_at_5 = self.in_top_k(top_5_indices.cpu(), x_ids)
        recall_at_10 = self.in_top_k(top_10_indices.cpu(), x_ids)
        recall_at_50 = self.in_top_k(top_50_indices.cpu(), x_ids)

        return recall_at_1, recall_at_5, recall_at_10, recall_at_50

    def cos_query(self, x, y):
        '''
        ===================================================================
        SF2F 논문 Table 2: Cosine Similarity 기반 Recall@K 계산
        ===================================================================
        
        📍 논문 핵심: "음성에서 올바른 얼굴을 찾는 정확도" 측정
        
        Args:
            x: 생성된 얼굴의 FaceNet 임베딩 (N_x, D)
            y: 실제 얼굴의 FaceNet 임베딩 (N_y, D)
            
        Returns:
            recall_tuple: (Recall@1, @2, @5, @10, @20, @50)
            
        📍 SF2F에서 가장 중요한 평가 지표:
        - Recall@1: 가장 유사한 1개가 정답인 비율 (엄격한 평가)
        - Recall@5: 상위 5개 중 정답이 있는 비율 (실용적 평가)  
        - Recall@10: 상위 10개 중 정답이 있는 비율 (관대한 평가)
        
        이는 음성-얼굴 매핑의 정확성을 직접 측정하는 핵심 메트릭입니다.
        '''
        # (N_x, D) --> (N_x, 1, D): 브로드캐스팅을 위한 차원 확장
        x = x.unsqueeze(1)
        # Ground Truth 레이블 생성: 각 쿼리의 정답 인덱스
        x_ids = torch.tensor(np.arange(x.shape[0])).cpu()
        # (N_y, D) --> (1, N_y, D): 브로드캐스팅을 위한 차원 확장
        y = y.unsqueeze(0)
        
        # 📍 논문 핵심: 코사인 유사도 계산 (방향성 고려)
        # Output: (N_x, N_y) - 각 생성 얼굴과 모든 실제 얼굴 간의 유사도
        cos_table = self.cos_sim(x, y)

        # 📍 SF2F 논문 Table 2: 다양한 K값에서의 Top-K 검색
        # 각 K값은 실제 응용에서의 다른 요구사항을 반영
        top_1_vals, top_1_indices = torch.topk(cos_table, 1, dim=-1)
        top_2_vals, top_2_indices = torch.topk(cos_table, 2, dim=-1)
        top_5_vals, top_5_indices = torch.topk(cos_table, 5, dim=-1)
        top_10_vals, top_10_indices = torch.topk(cos_table, 10, dim=-1)
        top_20_vals, top_20_indices = torch.topk(cos_table, 20, dim=-1)
        top_50_vals, top_50_indices = torch.topk(cos_table, 50, dim=-1)

        # 📍 각 K값에서 올바른 매칭 비율 계산
        recall_at_1 = self.in_top_k(top_1_indices.cpu(), x_ids)
        recall_at_2 = self.in_top_k(top_2_indices.cpu(), x_ids)
        recall_at_5 = self.in_top_k(top_5_indices.cpu(), x_ids)
        recall_at_10 = self.in_top_k(top_10_indices.cpu(), x_ids)
        recall_at_20 = self.in_top_k(top_20_indices.cpu(), x_ids)
        recall_at_50 = self.in_top_k(top_50_indices.cpu(), x_ids)

        recall_tuple = (recall_at_1, recall_at_2, recall_at_5, recall_at_10, \
            recall_at_20, recall_at_50)

        return recall_tuple

    def cal_ih_sim(self, x, y):
        '''
        ===================================================================
        SF2F 논문: Inter-Human Similarity Analysis
        ===================================================================
        
        📍 논문의 혁신적 평가 접근: 생성된 얼굴들 간의 다양성 측정
        
        기존 얼굴 임베딩 분포에 대해 사람 간 유사도를 계산합니다.
        이는 다음을 확인하기 위함입니다:
        1. Mode Collapse 방지: 모든 얼굴이 비슷하게 생성되지 않는가?
        2. 다양성 보존: 서로 다른 사람들의 고유한 특징이 유지되는가?
        3. 현실성: 생성된 얼굴들 간의 유사도가 실제 사람들과 비슷한가?

        Args:
            x: 첫 번째 임베딩 세트 (N_x, D)
            y: 두 번째 임베딩 세트 (N_y, D)
            
        Returns:
            ih_sim: 평균 inter-human similarity (0~1, 낮을수록 다양성이 높음)
        '''
        # 브로드캐스팅을 위한 차원 확장
        y = y.unsqueeze(0)  # (1, N_y, D)
        x = x.unsqueeze(1)  # (N_x, 1, D)
        
        # 📍 모든 쌍 간의 코사인 유사도 계산
        # Output: (N_x, N_y) - 모든 가능한 쌍 조합의 유사도
        cos_table = self.cos_sim(x, y)
        cos_table = cos_table.detach().cpu().numpy()
        
        # 📍 대각선 제외한 모든 쌍의 평균 유사도 계산
        # 자기 자신과의 유사도(대각선)는 제외하고 계산
        ih_sum = 0.0
        for i in range(cos_table.shape[0]):
            for j in range(cos_table.shape[1]):
                if i != j:  # 자기 자신 제외
                    ih_sum = ih_sum + cos_table[i, j]
                    
        # 평균 inter-human similarity 계산
        ih_sim = ih_sum / float(cos_table.shape[0] * (cos_table.shape[1] - 1))
        return ih_sim

    def in_top_k(self, top_k_indices, gt_labels):
        '''
        ===================================================================
        SF2F 논문: Top-K 검색에서 정답 포함 여부 확인
        ===================================================================
        
        📍 Recall@K의 핵심 계산 로직
        
        각 쿼리(생성된 얼굴)에 대해 Top-K 검색 결과에 
        정답(올바른 사람의 실제 얼굴)이 포함되어 있는지 확인합니다.
        
        Args:
            top_k_indices: Top-K 검색 결과 인덱스들
            gt_labels: Ground Truth 레이블 (정답 인덱스)
            
        Returns:
            recall_rate: 정답이 포함된 쿼리의 비율 (0~1)
        '''
        results = []
        for i, top_k_id in enumerate(top_k_indices):
            gt_label = gt_labels[i]  # i번째 쿼리의 정답
            
            # 📍 핵심: Top-K 결과에 정답이 있는가?
            if gt_label in top_k_id:
                results.append(1.0)  # 성공
            else:
                results.append(0.0)  # 실패
                
        # 전체 쿼리 중 성공한 비율 반환
        return np.mean(results)

    def get_metrics(self, model, recall_method='cos_sim', get_ih_sim=False):
        '''
        ===================================================================
        SF2F 논문: 종합적인 음성-얼굴 매핑 성능 평가
        ===================================================================
        
        📍 논문의 핵심 평가 메트릭들을 모두 계산하는 통합 함수
        
        SF2F 논문 Section 4.1에서 제시한 평가 지표들:
        1. Feature Space Distance (L1, L2): 특징 공간에서의 거리
        2. Cosine Similarity: 특징 벡터 간의 방향성 유사도  
        3. Identity Recall@K: 음성-얼굴 매핑 정확도 (가장 중요)
        4. Inter-Human Similarity: 다양성 및 현실성 평가
        
        Args:
            model: 평가할 SF2F 모델
            recall_method: 검색 방법 ('cos_sim' 또는 'L1')
            get_ih_sim: Inter-human similarity 계산 여부
            
        Returns:
            L2_dist, L1_dist, cos_sim, recall_tuple, [ih_sim]
        '''
        # 📍 1단계: 생성된 얼굴들의 FaceNet 임베딩 추출
        pred_img_embedding = self.get_pred_img_embeddings(model)
        
        # ===================================================================
        # SF2F 논문 Section 4.1: Feature Space Similarity Metrics
        # ===================================================================
        # 📍 L2 거리: 유클리드 거리 (일반적인 특징 공간 거리)
        L2_dist = self.L2_dist(self.dataset_embedding, pred_img_embedding)
        L2_dist = torch.mean(L2_dist).item()
        
        # 📍 L1 거리: 맨하탄 거리 (outlier에 robust)
        L1_dist = self.L1_dist(self.dataset_embedding, pred_img_embedding)
        L1_dist = torch.mean(L1_dist).item()
        
        # 📍 코사인 유사도: 방향성 고려한 유사도 (정규화된 특징에 적합)
        cos_sim = self.cos_sim(self.dataset_embedding, pred_img_embedding)
        cos_sim = torch.mean(cos_sim).item()

        # ===================================================================
        # SF2F 논문 Table 2: Identity Recall@K 계산
        # ===================================================================
        # 📍 평가 크기 설정: 논문에서 100, 200, 300개 샘플로 다양하게 평가
        if self.extraction_size is None:
            # 전체 데이터셋 사용
            pred_emb_to_use = pred_img_embedding
            data_emb_to_use = self.dataset_embedding
        elif isinstance(self.extraction_size, list):
            # 📍 다중 크기 평가: 더 robust한 성능 측정
            # 예: [100, 200, 300] → 100개, 100-200개, 200-300개로 나누어 평가
            pred_emb_to_use = [pred_img_embedding[0:self.extraction_size[0]], \
                pred_img_embedding[self.extraction_size[0]:self.extraction_size[1]], \
                pred_img_embedding[self.extraction_size[1]:self.extraction_size[2]]]
            data_emb_to_use = [self.dataset_embedding[0:self.extraction_size[0]], \
                self.dataset_embedding[self.extraction_size[0]:self.extraction_size[1]], \
                self.dataset_embedding[self.extraction_size[1]:self.extraction_size[2]]]
        else:
            # 지정된 크기만큼 사용
            pred_emb_to_use = pred_img_embedding[0:self.extraction_size]
            data_emb_to_use = self.dataset_embedding[0:self.extraction_size]

        # 📍 선택된 검색 방법으로 Recall@K 계산
        if recall_method == 'L1':
            # L1 거리 기반 검색
            if isinstance(pred_emb_to_use, list):
                # 다중 크기에서 평균 성능 계산
                recall_temp = []
                for i, pred_emb in enumerate(pred_emb_to_use):
                    recall_temp.append(self.L1_query(pred_emb, data_emb_to_use[i]))
                recall_tuple = tuple(np.mean(np.array(recall_temp), axis=0))
            else:
                recall_tuple = self.L1_query(pred_emb_to_use, data_emb_to_use)
                
        elif recall_method == 'cos_sim':
            # 📍 SF2F 논문에서 주로 사용하는 코사인 유사도 기반 검색
            if isinstance(pred_emb_to_use, list):
                # 다중 크기에서 평균 성능 계산
                recall_temp = []
                for i, pred_emb in enumerate(pred_emb_to_use):
                    recall_temp.append(self.cos_query(pred_emb, data_emb_to_use[i]))
                recall_tuple = tuple(np.mean(np.array(recall_temp), axis=0))
            else:
                recall_tuple = self.cos_query(pred_emb_to_use, data_emb_to_use)

        # ===================================================================
        # SF2F 논문: Inter-Human Similarity Analysis (선택적)
        # ===================================================================
        if get_ih_sim:
            # 📍 생성된 얼굴들 간의 다양성 측정
            ih_sim = self.cal_ih_sim(pred_img_embedding, self.dataset_embedding)
            return L2_dist, L1_dist, cos_sim, recall_tuple, ih_sim
        else:
            return L2_dist, L1_dist, cos_sim, recall_tuple

    def get_faces_from_different_segments(self, model, output_dir):
        '''
        ===================================================================
        SF2F 논문 Section 3: Multi-Modal Face Generation Analysis
        ===================================================================
        
        📍 논문 핵심: 다양한 음성 세그먼트에서 일관된 얼굴 생성 능력 평가
        
        이 함수는 SF2F 모델의 중요한 특성을 검증합니다:
        1. Temporal Consistency: 같은 사람의 다른 음성 세그먼트에서 일관된 얼굴 생성
        2. Voice-Face Stability: 음성의 시간적 변화에 robust한 얼굴 매핑
        3. Identity Preservation: 다양한 발화에서도 동일한 신원 유지
        
        📍 SF2F 논문에서 강조하는 핵심 평가 요소:
        - 음성의 prosody, 톤, 속도 변화에도 안정적인 얼굴 생성
        - 같은 사람이지만 다른 감정/상황의 음성에서도 일관성 유지
        - 시간에 따른 음성 변화에 대한 모델의 robustness 검증
        
        Args:
            model: 평가할 SF2F 모델
            output_dir: 결과 이미지들을 저장할 디렉토리
            
        Output Structure:
            output_dir/
            ├── 0/
            │   ├── origin_0.png, origin_1.png, ... (실제 얼굴들)
            │   └── pred_0.png, pred_1.png, ...   (생성된 얼굴들)
            ├── 1/
            │   └── ...
            └── ...
        '''
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        temp_loader = deepcopy(self.loader)
        
        # 📍 각 신원(ID)별로 다중 음성 세그먼트 처리
        for i in prog_bar(
            range(len(self.loader.dataset)),
            title="[S2fEvaluator: " + \
                "Generating Faces from Different Speech Segments]",
            width=50):
            ######### unpack the data #########
            # 📍 한 사람의 모든 얼굴 이미지 로드 (Ground Truth)
            imgs = temp_loader.dataset.get_all_faces_of_id(i)
            # 📍 한 사람의 모든 음성 세그먼트 로드 (다양한 발화)
            # SF2F 핵심: 서로 다른 음성에서 동일 인물 얼굴 생성 능력 테스트
            log_mels = temp_loader.dataset.get_all_mel_segments_of_id(i)
            imgs = imgs.cuda()
            log_mels = log_mels.type(self.float_dtype)
            ###################################
            
            # 📍 SF2F 모델로 여러 음성 세그먼트에서 얼굴 생성
            # 같은 사람의 다른 음성들 → 일관된 얼굴들이 생성되는가?
            with torch.no_grad():
                model_out = model(log_mels)
            imgs_pred, _ = model_out
            
            # Multi-Resolution 출력 처리 (가장 높은 해상도 사용)
            if isinstance(imgs_pred, tuple):
                imgs_pred = imgs_pred[-1]
                
            # 📍 이미지 후처리: 저장을 위한 정규화 해제
            # 실제 얼굴과 생성된 얼굴 모두 동일한 후처리 적용
            imgs = imagenet_deprocess_batch(
                imgs, normalize_method=self.image_normalize_method)
            imgs_pred = imagenet_deprocess_batch(
                imgs_pred, normalize_method=self.image_normalize_method)
            
            # 📍 각 신원별 디렉토리 생성
            identity_dir = os.path.join(output_dir, str(i))
            gfile.MkDir(identity_dir)

            # 📍 실제 얼굴 이미지들 저장 (비교 기준)
            for j in range(imgs.shape[0]):
                img_np = imgs[j].numpy().transpose(1, 2, 0)  # CHW → HWC
                img_path = os.path.join(identity_dir, 'origin_%d.png' % j)
                imwrite(img_path, img_np)

            # 📍 생성된 얼굴 이미지들 저장 (SF2F 결과)
            # 각 이미지는 서로 다른 음성 세그먼트에서 생성됨
            # 논문 평가: 이들이 얼마나 일관되고 실제 얼굴과 유사한가?
            for k in range(imgs_pred.shape[0]):
                img_np = imgs_pred[k].numpy().transpose(1, 2, 0)  # CHW → HWC
                img_path = os.path.join(identity_dir, 'pred_%d.png' % k)
                imwrite(img_path, img_np)

    def L2_distances(self, x, y=None):
        '''
        ===================================================================
        SF2F 논문 Section 4.1: Feature Space Distance Matrix 계산
        ===================================================================
        
        📍 논문에서 사용하는 유클리드 거리 기반 유사도 매트릭스 계산
        
        이 함수는 SF2F 평가에서 다음과 같이 활용됩니다:
        1. 생성된 얼굴들 간의 거리 분포 분석
        2. 실제 얼굴과 생성된 얼굴 간의 거리 측정
        3. Inter-Human Distance Analysis의 기반 계산
        4. Clustering Analysis를 위한 거리 매트릭스 제공
        
        Args:
            x: 첫 번째 임베딩 세트 (Nxd 매트릭스)
            y: 두 번째 임베딩 세트 (Mxd 매트릭스, 선택적)
               None이면 x 내의 모든 쌍 간 거리 계산
               
        Returns:
            dist: NxM 거리 매트릭스
                  dist[i,j] = ||x[i,:] - y[j,:]||^2 (제곱 유클리드 거리)
                  
        📍 SF2F에서 활용 예시:
        - 생성 품질 분석: 생성된 얼굴들이 얼마나 다양한가?
        - 매핑 정확도: 생성된 얼굴이 실제 얼굴과 얼마나 가까운가?
        - 클러스터링: 비슷한 음성 특징을 가진 사람들의 얼굴 그룹 분석
        '''
        if y is not None:
            # 📍 서로 다른 두 세트 간의 거리 계산
            # x: (N, d), y: (M, d) → distances: (N, M)
            differences = x.unsqueeze(1) - y.unsqueeze(0)  # Broadcasting
        else:
            # 📍 같은 세트 내의 모든 쌍 간 거리 계산 (자기 자신 포함)
            # x: (N, d) → distances: (N, N)
            differences = x.unsqueeze(1) - x.unsqueeze(0)  # Broadcasting
            
        # 📍 제곱 유클리드 거리 계산: ||a - b||^2 = sum((a - b)^2)
        # SF2F에서 특징 공간의 거리 측정에 표준적으로 사용
        distances = torch.sum(differences * differences, -1)
        return distances
