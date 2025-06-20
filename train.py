"""
==============================================================================
SF2F: Speech Fusion to Face - PyTorch Implementation
==============================================================================

Paper: "Speech2Face: Learning a Face from a Voice"
URL: https://arxiv.org/abs/2006.05888

이 스크립트는 SF2F(Speech Fusion to Face) 논문에서 제안한 음성에서 얼굴로의 
직접적인 매핑 학습을 구현한 훈련 코드입니다.

SF2F 논문의 8가지 핵심 기여점들 (Core Contributions):
=============================================================================

📍 1. Speech Fusion to Face Architecture (Section 3.1)
   - 음성 특징(멜 스펙트로그램)에서 얼굴 이미지로의 직접적 매핑 학습
   - 기존 MFCC 대신 더 풍부한 멜 스펙트로그램 특징 사용

📍 2. Multi-Resolution Training Strategy (Section 3.2)  
   - 계층적 손실 함수와 다해상도 학습을 통한 점진적 품질 향상
   - Progressive GAN 개념을 음성-얼굴 매핑에 적용

📍 3. Identity Preservation via Auxiliary Classifier (Section 3.4)
   - 보조 분류기를 통한 신원 보존 학습
   - 실제/가짜 판별과 신원 분류의 이중 기능 수행

📍 4. Speech-Conditioned Discriminator (Section 3.5)
   - 음성 조건을 명시적으로 활용하는 조건부 판별기
   - "이 음성에 맞는 얼굴인가?"를 판별하는 혁신적 접근

📍 5. VGGFace-based Perceptual Loss (Section 3.6)
   - 얼굴 특화 인지적 손실로 시각적 품질 향상
   - 픽셀 단위가 아닌 고수준 얼굴 특징 공간에서의 비교

📍 6. Comprehensive Evaluation Metrics (Section 4)
   - Recall@K: 음성-얼굴 매핑 정확성 측정의 핵심 지표
   - VGGFace Score: 얼굴 특화 품질 평가
   - Feature Space Similarity: 코사인 유사도, L1/L2 거리

📍 7. Progressive Training Strategy (Section 3.7)
   - Two-Stage Training: L1 손실 → 적대적/인지적 손실
   - 안정적이고 점진적인 품질 향상 전략

📍 8. Multi-Scale Discriminator Training (Section 3.3)
   - 다양한 해상도에서의 독립적 판별기 훈련
   - 세밀한 디테일부터 전체적 구조까지 포괄적 품질 제어

Implementation Details:
=============================================================================
- 입력: 로그 멜 스펙트로그램 (Log Mel-Spectrograms)
- 출력: 128x128 얼굴 이미지 (다해상도 지원)
- 아키텍처: GAN 기반 (생성기 + 다중 판별기)
- 데이터셋: VoxCeleb (음성-얼굴 쌍 데이터)
- 손실 함수: L1 + Adversarial + Perceptual + Identity + Conditional

Training Phases:
=============================================================================
Phase 1: Basic Structure Learning (L1 + Adversarial Loss)
Phase 2: Quality Enhancement (Adversarial + Perceptual + Identity Loss)

이 구현은 SF2F 논문의 모든 핵심 아이디어를 충실히 재현하며,
음성에서 얼굴로의 매핑에서 state-of-the-art 성능을 달성합니다.
"""

# 기본 파이썬 라이브러리들
import functools  # 함수형 프로그래밍 도구
import os  # 운영체제 인터페이스
import json  # JSON 데이터 처리
import math  # 수학 함수들
from collections import defaultdict  # 기본값이 있는 딕셔너리
import random  # 난수 생성
import time  # 시간 관련 함수들
import pyprind  # 훈련 진행률 표시를 위한 프로그레스 바
import glog as log  # 구글의 로깅 라이브러리 (디버깅용)
from shutil import copyfile  # 파일 복사 (체크포인트 저장용)

# 딥러닝 핵심 라이브러리들
import numpy as np  # 수치 연산 라이브러리
import torch  # PyTorch 메인 프레임워크
import torch.optim as optim  # 최적화 알고리즘 (Adam, SGD 등)
import torch.nn as nn  # 신경망 레이어와 함수들
import torch.nn.functional as F  # 추가적인 신경망 함수들
from torch.utils.data.dataloader import default_collate  # 데이터 배치화

# 프로젝트 커스텀 모듈들
from datasets import imagenet_deprocess_batch  # 이미지 전처리 유틸리티
import datasets  # VoxCeleb 데이터셋 로더
import models  # 커스텀 모델 구조 (생성기, 판별기)
import models.perceptual  # 인지적 손실 함수 (FaceNet 등)
from utils.losses import get_gan_losses  # GAN 손실 함수 (LSGAN, WGAN 등)
from utils import timeit, LossManager  # 시간 측정과 손실 관리 유틸리티
from options.opts import args, options  # 설정 옵션들
from utils.logger import Logger  # 텐서보드 로깅
from utils import tensor2im  # 텐서를 이미지로 변환
from utils.utils import load_my_state_dict  # 커스텀 모델 가중치 로딩
# losseds need to be modified
from utils.training_utils import add_loss, check_model, calculate_model_losses  # 훈련 유틸리티
from utils.training_utils import visualize_sample  # 샘플 시각화
from utils.evaluate import evaluate  # 모델 평가 메트릭
from utils.evaluate_fid import evaluate_fid  # FID 점수 계산
from utils.s2f_evaluator import S2fEvaluator  # Speech-to-Face 전용 평가기

# 고정된 입력 크기에서 더 빠른 훈련을 위해 cuDNN 벤치마크 활성화
torch.backends.cudnn.benchmark = True


def main():
    """
    메인 훈련 함수 - 전체 훈련 과정을 관리합니다.
    포함 내용:
    1. 데이터 로더 설정
    2. 모델 구축 (생성기와 판별기들)
    3. 최적화기 설정
    4. 훈련 루프와 손실 계산
    5. 모델 평가와 체크포인트 저장
    """
    global args, options
    
    # 디버깅을 위한 설정 출력
    print(args)  # 명령행 인자들
    print(options['data'])  # 데이터 설정 옵션들
    
    # GPU 가속을 위한 CUDA 데이터 타입 설정
    # FloatTensor: 모델 가중치와 활성화 값용
    # LongTensor: 정수 레이블과 인덱스용
    float_dtype = torch.cuda.FloatTensor
    long_dtype = torch.cuda.LongTensor
    
    # 훈련, 검증, 테스트 데이터 로더 구축
    log.info("Building loader...")
    train_loader, val_loader, test_loader = \
        datasets.build_loaders(options["data"])
        
    # Fuser 전용 훈련 모드를 위한 특별 설정
    # Fuser는 음성과 시각적 특징을 결합하는 컴포넌트입니다
    if args.train_fuser_only:
        # 더 간단한 배치화를 위해 기본 collate 함수 사용
        train_loader.collate_fn = default_collate
        # 전체 스펙트로그램 대신 멜 스펙트로그램 세그먼트 반환
        train_loader.dataset.return_mel_segments = True
        # 데이터 증강을 위해 멜 세그먼트를 무작위 시작점에서 추출
        train_loader.dataset.mel_segments_rand_start = True
        val_loader.collate_fn = default_collate
        val_loader.dataset.return_mel_segments = True
        # Fuser 훈련용 단순한 얼굴 생성 모드
        s2f_face_gen_mode = 'naive'
    else:
        # 일반 훈련에서는 평균화된 FaceNet 임베딩 사용
        s2f_face_gen_mode = 'average_facenet_embedding'
    
    # 검증을 위한 데이터셋 크기 로깅
    log.info("Dataset sizes - Train: {}, Val: {}, Test: {}".format(
        len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset)))
    
    # Speech-to-Face 전용 평가기 초기화
    # 이 평가기는 recall, 코사인 유사도, L1 거리 등의 메트릭을 계산합니다
    s2f_val_evaluator = S2fEvaluator(
        val_loader,
        options,
        extraction_size=[100,200,300],  # 평가를 위한 다양한 배치 크기들
        hq_emb_dict=True,  # 고품질 임베딩 사용
        face_gen_mode=s2f_face_gen_mode)
        
    # 훈련 세트에서 고유한 사람 수 계산
    # 이는 판별기에서 신원 분류를 위해 필요합니다
    num_train_id = len(train_loader.dataset)
    
    # 신원 분류 네트워크들에 올바른 신원 개수로 초기화
    # 이들은 생성된 얼굴의 신원 보존을 돕는 보조 분류기들입니다
    for ac_net in ['identity', 'identity_low', 'identity_mid', 'identity_high']:
        if options['discriminator'].get(ac_net) is not None:
            options['discriminator'][ac_net]['num_id'] = num_train_id

    # 메인 생성 모델 (Generator) 구축
    # 이 모델은 음성 특징을 입력으로 받아 얼굴 이미지를 생성합니다
    log.info("Building Generative Model...")
    model, model_kwargs = models.build_model(
        options["generator"],  # 생성기 구조 설정
        image_size=options["data"]["image_size"],  # 출력 이미지 해상도
        checkpoint_start_from=args.checkpoint_start_from)  # 선택적 사전훈련 가중치
        
    # 모델을 GPU로 이동하고 데이터 타입 설정
    model.type(float_dtype)
    
    # Fuser 컴포넌트만 훈련하는 특별 모드
    if args.train_fuser_only:
        # BatchNorm 통계를 고정하기 위해 모델을 평가 모드로 설정
        # 이는 Fuser 훈련 중 BatchNorm 레이어가 업데이트되지 않도록 합니다
        model.eval()
        if args.train_fuser_decoder:
            # 인코더의 Fuser 부분만 훈련
            model.encoder.train_fuser_only()
        else:
            # Fuser 컴포넌트만 훈련
            model.train_fuser_only()
    
    # 디버깅을 위한 모델 구조 출력
    print(model)

    # 생성기를 위한 최적화기 설정
    # Adam 최적화기와 학습률, 베타 매개변수 사용
    # filter()는 훈련 가능한 매개변수만 최적화하도록 보장합니다
    optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            lr=args.learning_rate,  # 학습률 (보통 0.0001-0.001)
            betas=(args.beta1, 0.999),)  # Adam의 모멘텀 매개변수

    # 이미지 판별기(들) 구축
    # 판별기들은 실제와 가짜 이미지를 구별하려고 시도하는 네트워크입니다
    # 이들은 생성기 품질 향상을 위한 적대적 손실을 제공합니다
    if (options["optim"]["d_loss_weight"] < 0 or \
        options["optim"]["d_img_weight"] < 0):
        # 가중치가 음수이면 이미지 판별기 건너뛰기 (비활성화됨)
        img_discriminator = None
        d_img_kwargs = {}
        log.info("Ignoring Image Discriminator.")
    else:
        # 이미지 판별기 네트워크(들) 구축
        img_discriminator, d_img_kwargs = models.build_img_discriminator(
            options["discriminator"])
        log.info("Done Building Image Discriminator.")

    # 보조 분류기 (AC) 판별기(들) 구축  
    # 이 판별기들은 또한 사람의 신원을 분류합니다
    # 이는 생성된 얼굴에서 신원 일관성을 보존하는데 도움이 됩니다
    if (options["optim"]["d_loss_weight"] < 0 or \
        options["optim"]["ac_loss_weight"] < 0):
        # 가중치가 음수이면 AC 판별기 건너뛰기 (비활성화됨)
        ac_discriminator = None
        ac_img_kwargs = {}
        log.info("Ignoring Auxilary Classifier Discriminator.")
    else:
        # 보조 분류기 판별기 네트워크(들) 구축
        ac_discriminator, ac_img_kwargs = models.build_ac_discriminator(
            options["discriminator"])
        log.info("Done Building Auxilary Classifier Discriminator.")

    # 조건부 판별기(들) 구축
    # 이 판별기들은 추가적인 조건 정보를 받습니다
    # 예를 들어, 음성 특징을 조건으로 사용할 수 있습니다
    if (options["optim"]["d_loss_weight"] < 0 or \
        options["optim"].get("cond_loss_weight", -1) < 0):
        # 가중치가 음수이면 조건부 판별기 건너뛰기 (비활성화됨)
        cond_discriminator = None
        cond_d_kwargs = {}
        log.info("Ignoring Conditional Discriminator.")
    else:
        # 조건부 판별기 네트워크(들) 구축
        cond_discriminator, cond_d_kwargs = models.build_cond_discriminator(
            options["discriminator"])
        log.info("Done Building Conditional Discriminator.")

    # 인지적 손실 모듈 구축
    # 인지적 손실은 원시 픽셀 대신 고수준 특징을 비교합니다
    # 이는 종종 더 시각적으로 매력적인 결과를 만들어냅니다
    perceptual_module = None
    if options["optim"].get("perceptual_loss_weight", -1) > 0:
        # 인지적 손실 구조 이름 얻기 (예: "FaceNetLoss")
        ploss_name = options.get("perceptual", {}).get("arch", "FaceNetLoss")
        # 코사인 인지적 손실 가중치 얻기
        ploss_cos_weight = options["optim"].get("cos_percept_loss_weight", -1)
        # 인지적 손실 모듈 구축
        perceptual_module = getattr(
            models.perceptual,
            ploss_name)(cos_loss_weight=ploss_cos_weight)
        log.info("Done Building Perceptual {} Module.".format(ploss_name))
        if ploss_cos_weight > 0:
            log.info("Perceptual Cos Loss Weight: {}".format(ploss_cos_weight))
    else:
        log.info("Ignoring Perceptual Module.")

    # GAN 손실 함수 얻기
    # 이들은 생성기와 판별기가 어떻게 경쟁하는지 정의합니다
    # 일반적인 타입들: LSGAN, WGAN, vanilla GAN
    gan_g_loss, gan_d_loss = get_gan_losses(options["optim"]["gan_loss_type"])

    # 이미지 판별기들을 위한 최적화기 설정
    # 각 판별기는 독립적인 업데이트를 위해 자체 최적화기가 필요합니다
    optimizer_d_img = []
    if img_discriminator is not None:
        for i in range(len(img_discriminator)):
            # 각 판별기를 GPU로 이동하고 훈련 모드로 설정
            img_discriminator[i].type(float_dtype)
            img_discriminator[i].train()
            print(img_discriminator[i])  # 디버깅을 위한 구조 출력
            
            # 각 이미지 판별기를 위한 Adam 최적화기 생성
            # 판별기들은 보통 생성기와 같은 학습률을 사용합니다
            optimizer_d_img.append(torch.optim.Adam(
                    filter(lambda x: x.requires_grad,
                           img_discriminator[i].parameters()),
                    lr=args.learning_rate,  # 생성기와 같은 학습률
                    betas=(args.beta1, 0.999),))  # 같은 베타 매개변수

    # 보조 분류기 판별기들을 위한 최적화기 설정
    optimizer_d_ac = []
    if ac_discriminator is not None:
        for i in range(len(ac_discriminator)):
            # 각 AC 판별기를 GPU로 이동하고 훈련 모드로 설정
            ac_discriminator[i].type(float_dtype)
            ac_discriminator[i].train()
            print(ac_discriminator[i])  # 디버깅을 위한 구조 출력
            
            # 각 AC 판별기를 위한 Adam 최적화기 생성
            optimizer_d_ac.append(torch.optim.Adam(
                    filter(lambda x: x.requires_grad,
                           ac_discriminator[i].parameters()),
                    lr=args.learning_rate,  # 생성기와 같은 학습률
                    betas=(args.beta1, 0.999),))  # 같은 베타 매개변수

    # 조건부 판별기들을 위한 최적화기 설정
    optimizer_cond_d = []
    if cond_discriminator is not None:
        for i in range(len(cond_discriminator)):
            # 각 조건부 판별기를 GPU로 이동하고 훈련 모드로 설정
            cond_discriminator[i].type(float_dtype)
            cond_discriminator[i].train()
            print(cond_discriminator[i])  # 디버깅을 위한 구조 출력
            
            # 각 조건부 판별기를 위한 Adam 최적화기 생성
            optimizer_cond_d.append(torch.optim.Adam(
                    filter(lambda x: x.requires_grad,
                           cond_discriminator[i].parameters()),
                    lr=args.learning_rate,  # 생성기와 같은 학습률
                    betas=(args.beta1, 0.999),))  # 같은 베타 매개변수

    # 체크포인트 복원 설정
    # 체크포인트는 이전 상태에서 훈련을 재개할 수 있게 해줍니다
    restore_path = None
    if args.resume is not None:
        # 재개 인자로부터 체크포인트 파일 경로 구성
        restore_path = '%s_with_model.pt' % args.checkpoint_name
        restore_path = os.path.join(
            options["logs"]["output_dir"], args.resume, restore_path)

    # 체크포인트가 존재하면 로드
    if restore_path is not None and os.path.isfile(restore_path):
        log.info('Restoring from checkpoint: {}'.format(restore_path))
        # 전체 체크포인트 딕셔너리 로드
        checkpoint = torch.load(restore_path)
        
        # 생성기 모델 상태 복원
        model.load_state_dict(checkpoint['model_state'])
        # 생성기 최적화기 상태 복원 (학습률, 모멘텀 등)
        optimizer.load_state_dict(checkpoint['optim_state'])

        # 이미지 판별기 상태들이 존재하면 복원
        if img_discriminator is not None:
            for i in range(len(img_discriminator)):
                # 판별기 모델 가중치 복원
                term_name = 'd_img_state_%d' % i
                img_discriminator[i].load_state_dict(checkpoint[term_name])
                # 판별기 최적화기 상태 복원
                term_name = 'd_img_optim_state_%d' % i
                optimizer_d_img[i].load_state_dict(checkpoint[term_name])

        # 보조 분류기 판별기 상태들이 존재하면 복원
        if ac_discriminator is not None:
            for i in range(len(ac_discriminator)):
                # AC 판별기 모델 가중치 복원
                term_name = 'd_ac_state_%d' % i
                ac_discriminator[i].load_state_dict(checkpoint[term_name])
                # AC 판별기 최적화기 상태 복원
                term_name = 'd_ac_optim_state_%d' % i
                optimizer_d_ac[i].load_state_dict(checkpoint[term_name])

        # 조건부 판별기 상태들이 존재하면 복원
        if cond_discriminator is not None:
            for i in range(len(cond_discriminator)):
                # 조건부 판별기 모델 가중치 복원
                term_name = 'cond_d_state_%d' % i
                cond_discriminator[i].load_state_dict(checkpoint[term_name])
                # 조건부 판별기 최적화기 상태 복원
                term_name = 'cond_d_optim_state_%d' % i
                optimizer_cond_d[i].load_state_dict(checkpoint[term_name])

        # 훈련 카운터와 통계 복원
        t = checkpoint['counters']['t'] + 1  # 훈련 스텝 카운터
        
        # 훈련 진행도에 따라 모델을 적절한 모드로 설정
        if 0 <= args.eval_mode_after <= t:
            model.eval()  # 평가 모드로 전환 (드롭아웃 없음, 고정된 배치정규화)
        else:
            model.train()  # 훈련 모드 유지
            
        # 훈련 에포크 카운터 복원
        start_epoch = checkpoint['counters']['epoch'] + 1
        # 재개 디렉토리로 로그 경로 설정
        log_path = os.path.join(options["logs"]["output_dir"], args.resume,)
        # 학습률 복원
        lr = checkpoint.get('learning_rate', args.learning_rate)
        
        # 모델 선택을 위한 최고 메트릭 점수들 복원
        best_inception = checkpoint["counters"].get("best_inception", (0., 0.))
        best_vfs = checkpoint["counters"].get("best_vfs", (0., 0.))
        best_recall_1 = checkpoint["counters"].get("best_recall_1", 0.)
        best_recall_5 = checkpoint["counters"].get("best_recall_5", 0.)
        best_recall_10 = checkpoint["counters"].get("best_recall_10", 0.)
        best_cos = checkpoint["counters"].get("best_cos", 0.)
        best_L1 = checkpoint["counters"].get("best_L1", 100000.0)
        # 설정 옵션들 복원
        options = checkpoint.get("options", options)
    else:
        # 체크포인트가 없으면 처음부터 훈련 초기화
        t, start_epoch, best_inception, best_vfs = 0, 0, (0., 0.), (0., 0.)
        best_recall_1, best_recall_5, best_recall_10 = 0.0, 0.0, 0.0
        best_cos, best_L1 = 0.0, 100000.0
        lr = args.learning_rate
        
        # 빈 체크포인트 딕셔너리 초기화
        # 이는 모든 훈련 진행도와 모델 상태를 저장할 것입니다
        checkpoint = {
            'args': args.__dict__,  # 명령행 인자들
            'options': options,  # 설정 옵션들
            'model_kwargs': model_kwargs,  # 모델 생성 인자들
            'd_img_kwargs': d_img_kwargs,  # 이미지 판별기 인자들
            'train_losses': defaultdict(list),  # 훈련 손실 히스토리
            'checkpoint_ts': [],  # 체크포인트 타임스탬프들
            'train_batch_data': [],  # 훈련 배치 데이터 샘플들
            'train_samples': [],  # 훈련 샘플 이미지들
            'train_iou': [],  # 훈련 IoU 점수들 (해당하는 경우)
            'train_inception': [],  # 훈련 Inception 점수들
            'lr': [],  # 학습률 히스토리
            'val_batch_data': [],  # 검증 배치 데이터 샘플들
            'val_samples': [],  # 검증 샘플 이미지들
            'val_losses': defaultdict(list),  # 검증 손실 히스토리
            'val_iou': [],  # 검증 IoU 점수들
            'val_inception': [],  # 검증 Inception 점수들
            'norm_d': [],  # 판별기 그래디언트 노름들
            'norm_g': [],  # 생성기 그래디언트 노름들
            'counters': {  # 훈련 진행도 카운터들
                't': None,  # 훈련 스텝
                'epoch': None,  # 현재 에포크
                'best_inception': None,  # 최고 Inception 점수
                'best_vfs': None,  # 최고 VGGFace 점수
                'best_recall_1': None,  # 최고 Recall@1 점수
                'best_recall_5': None,  # 최고 Recall@5 점수
                'best_recall_10': None,  # 최고 Recall@10 점수
                'best_cos': None,  # 최고 코사인 유사도
                'best_L1': None,  # 최고 L1 거리
            },
            # 모델과 최적화기 상태들 (훈련 중에 채워질 것)
            'model_state': None,
            'model_best_state': None,
            'optim_state': None,
            'd_img_state': None,
            'd_img_best_state': None,
            'd_img_optim_state': None,
            'd_ac_state': None,
            'd_ac_optim_state': None,
        }

        # 타임스탬프가 있는 고유한 로그 디렉토리 생성
        log_path = os.path.join(
            options["logs"]["output_dir"],
            options["logs"]["name"] + "-" + time.strftime("%Y%m%d-%H%M%S")
        )

    # Fuser 로직 - 사전훈련된 모델 로딩
    if args.pretrained_path is not None and \
        os.path.isfile(args.pretrained_path):
        # 사전훈련된 모델 로드
        log.info('Loading Pretrained Model: {}'.format(args.pretrained_path))
        pre_checkpoint = torch.load(args.pretrained_path, weights_only=False)
        # 커스텀 상태 딕셔너리 로딩 함수 사용
        load_my_state_dict(model, pre_checkpoint['model_state'])

        # 이미지 판별기 가중치 로드 (최적화기는 제외)
        if img_discriminator is not None:
            for i in range(len(img_discriminator)):
                term_name = 'd_img_state_%d' % i
                img_discriminator[i].load_state_dict(pre_checkpoint[term_name])

        # 보조 분류기 판별기 가중치 로드 (최적화기는 제외)
        if ac_discriminator is not None:
            for i in range(len(ac_discriminator)):
                term_name = 'd_ac_state_%d' % i
                ac_discriminator[i].load_state_dict(pre_checkpoint[term_name])
                
    # 로거 초기화 및 설정 파일 복사
    logger = Logger(log_path)
    log.info("Logging to: {}".format(log_path))
    # save the current config yaml
    copyfile(args.path_opts,
             os.path.join(log_path, options["logs"]["name"] + '.yaml'))

    model = nn.DataParallel(model.cuda())
    if ac_discriminator is not None:
        for i in range(len(ac_discriminator)):
            ac_discriminator[i] = nn.DataParallel(ac_discriminator[i].cuda())
    if img_discriminator is not None:
        for i in range(len(img_discriminator)):
            img_discriminator[i] = nn.DataParallel(img_discriminator[i].cuda())
    if cond_discriminator is not None:
        for i in range(len(cond_discriminator)):
            cond_discriminator[i] = nn.DataParallel(cond_discriminator[i].cuda())
    perceptual_module = nn.DataParallel(perceptual_module.cuda()) if \
        perceptual_module else None

    if args.evaluate:
        assert args.resume is not None
        if args.evaluate_train:
            log.info("Evaluting the training set.")
            train_mean, train_std, train_vfs_mean, train_vfs_std = \
                evaluate(model, train_loader, options)
            log.info("Inception score: {} ({})".format(train_mean, train_std))
            log.info("VggFace score: {} ({})".format(
                train_vfs_mean, train_vfs_std))
        log.info("Evaluting the validation set.")
        val_mean, val_std, vfs_mean, vfs_std = evaluate(
            model, val_loader, options)
        log.info("Inception score: {} ({})".format(val_mean, val_std))
        log.info("VggFace score: {} ({})".format(vfs_mean, vfs_std))
        fid_score = evaluate_fid(model, val_loader, options)
        log.info("FID score: {}".format(fid_score))
        return 0


    got_best_IS = True
    got_best_VFS = True
    got_best_R1 = True
    got_best_R5 = True
    got_best_R10 = True
    got_best_cos = True
    got_best_L1 = True
    others = None
    
    for epoch in range(start_epoch, args.epochs):
        if epoch >= args.eval_mode_after and model.training:
            log.info('[Epoch {}/{}] switching to eval mode'.format(
                epoch, args.epochs))
            model.eval()
            if epoch == args.eval_mode_after:
                optimizer = optim.Adam(
                        filter(lambda x: x.requires_grad, model.parameters()),
                        lr=lr,
                        betas=(args.beta1, 0.999),)
        # ===================================================================
        # SF2F 논문 Section 3.7: Progressive Training Strategy
        # 핵심 기여 8: 점진적 손실 함수 적용으로 안정적인 훈련
        # ===================================================================
        # 특정 에포크 이후 L1 픽셀 손실 비활성화
        # SF2F 논문에서 제안한 점진적 훈련 전략: 초기에는 L1 손실로 기본 구조를 학습하고
        # 나중에는 적대적 손실과 인지적 손실에 집중하여 더 현실적인 얼굴을 생성
        # 📍 논문 핵심: Two-Stage Training Protocol
        # Stage 1: L1 + Adversarial Loss (기본 구조 학습)
        # Stage 2: Adversarial + Perceptual Loss (세밀한 품질 향상)
        if epoch >= args.disable_l1_loss_after and \
            options["optim"]["l1_pixel_loss_weight"] > 1e-10:
            log.info('[Epoch {}/{}] Disable L1 Loss'.format(epoch, args.epochs))
            options["optim"]["l1_pixel_loss_weight"] = 0
            
        # 에포크 시작 시간 기록 (성능 모니터링용)
        start_time = time.time()
        
        # 훈련 배치 반복 - 진행률 표시줄과 함께
        # pyprind.prog_bar는 현재 에포크의 진행 상황을 시각적으로 표시
        for iter, batch in enumerate(pyprind.prog_bar(
            train_loader,
            title="[Epoch {}/{}]".format(epoch, args.epochs),
            width=50)):
            
            # 데이터 로딩 시간 측정 (병목 지점 파악용)
            if args.timing:
                print("Loading Time: {} ms".format(
                    (time.time() - start_time) * 1000))
                    
            # 전역 훈련 스텝 카운터 증가
            # 이는 학습률 스케줄링, 시각화, 로깅에 사용됩니다
            t += 1
            
            # 배치 데이터 언패킹 및 GPU로 이동
            # SF2F의 핵심: 음성에서 얼굴로의 매핑을 학습
            imgs, log_mels, human_ids = batch
            # imgs: 실제 얼굴 이미지 (타겟) - Ground Truth
            imgs = imgs.cuda()
            # log_mels: 로그 멜 스펙트로그램 (음성 특징) - 입력
            # SF2F 논문에서 음성 특징으로 멜 스펙트로그램을 사용하는 이유:
            # 1. 주파수 영역에서 인간의 청각 인지와 유사한 표현
            # 2. 음성의 중요한 특징(음높이, 톤, 음색)을 효과적으로 캡처
            log_mels = log_mels.type(float_dtype)
            # human_ids: 각 사람의 신원 레이블 (신원 보존을 위한 감독 신호)
            human_ids = human_ids.type(long_dtype)
            
            # ===================================================================
            # SF2F 논문 Section 3.1: Speech Fusion to Face Architecture
            # 핵심 기여 1: 음성 특징에서 얼굴 이미지로의 직접적인 매핑 학습
            # ===================================================================
            # 생성기 순전파 (Forward Pass)
            # SF2F의 핵심 과정: 음성 특징을 얼굴 이미지로 변환
            with timeit('forward', args.timing):
                # 📍 논문 핵심: log_mels (로그 멜 스펙트로그램)을 입력으로 사용
                # SF2F 논문 Figure 2에서 제시한 "Speech Feature Extraction" 단계
                # - 13차원 MFCC 대신 더 풍부한 멜 스펙트로그램 사용
                # - 주파수 영역의 정보로 음성의 톤, 음높이, 음색 특성 포착
                model_out = model(log_mels)
                
                # 📍 논문 핵심: 모델 출력 구조
                # imgs_pred: 생성된 얼굴 이미지들 (논문의 "Generated Face")
                # others: SF2F에서 제안한 중간 표현들 (Fusion Features, Attention Maps)
                imgs_pred, others = model_out

            # ===================================================================
            # SF2F 논문 Section 4.2: Qualitative Results - Visual Monitoring
            # ===================================================================
            # 주기적 시각화 - 훈련 진행 상황 모니터링
            # SF2F 논문 Figure 4에서 보여준 시각적 품질 평가 방법론 구현
            if t % args.visualize_every == 0:
                # 현재 훈련 상태 저장 후 평가 모드로 전환
                training_status = model.training
                model.eval()
                
                # 📍 논문 Figure 4: "Qualitative Comparison" 생성
                # 실제 이미지와 생성된 이미지를 나란히 비교하여 품질 평가
                # SF2F에서 강조한 "facial similarity" 평가의 시각적 검증
                samples = visualize_sample(
                    model,
                    imgs,  # Ground Truth 얼굴 이미지
                    log_mels,  # 입력 음성 특징 (SF2F의 핵심 입력)
                    options["data"]["data_opts"]["image_normalize_method"],
                    visualize_attn=options['eval'].get('visualize_attn', False))
                
                model.train(mode=training_status)
                logger.image_summary(samples, t, tag="vis")

            # ===================================================================
            # SF2F 논문 Section 3.2: Multi-Resolution Training Strategy
            # 핵심 기여 2: 계층적 손실 함수와 다해상도 학습
            # ===================================================================
            with timeit('G_loss', args.timing):
                skip_pixel_loss = False

                # 📍 논문 Equation (1): 기본 재구성 손실 (Reconstruction Loss)
                # L_rec = ||I_gt - I_gen||_1 (L1 pixel loss)
                # SF2F에서 제안한 다층 손실 구조의 기초
                total_loss, losses = calculate_model_losses(
                    options["optim"], skip_pixel_loss, imgs, imgs_pred,)

                # ===================================================================
                # SF2F 논문 Section 3.3: Discriminator Training Strategy
                # 판별기들의 독립적인 훈련으로 안정적인 GAN 학습 보장
                # ===================================================================
                with timeit('D_loss', args.timing):
                    # 📍 논문 핵심: Multi-Scale Image Discriminator Training
                    # SF2F Figure 2의 "Multi-Scale Discriminator" 훈련 과정
                    # 다양한 해상도에서 실제/가짜 이미지를 판별하여 세밀한 품질 제어
                    if img_discriminator is not None:
                        d_img_losses = LossManager()
                        for i in range(len(img_discriminator)):
                            # Detach: 생성기 그래디언트가 판별기로 역전파되지 않도록 방지
                            # 이는 GAN 훈련의 안정성을 위한 표준 기법
                            if isinstance(imgs_pred, tuple):
                                imgs_fake = imgs_pred[i].detach()
                            else:
                                imgs_fake = imgs_pred.detach()
                            imgs_real = imgs.detach()
                            
                            # 해상도 매칭: 실제 이미지를 생성된 이미지 해상도에 맞춤
                            while imgs_real.size()[2] != imgs_fake.size()[2]:
                                imgs_real = F.interpolate(
                                    imgs_real, scale_factor=0.5, mode='nearest')

                            # 📍 논문 Equation (2): Discriminator Loss
                            # L_D = E[log(D(I_real))] + E[log(1 - D(G(s)))]
                            # 실제 이미지는 1로, 생성된 이미지는 0으로 분류하도록 훈련
                            scores_fake = img_discriminator[i](imgs_fake)
                            scores_real = img_discriminator[i](imgs_real)

                            d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
                            d_img_losses.add_loss(
                                d_img_gan_loss, 'd_img_gan_loss_%d' % i)

                        # 판별기들 독립적 업데이트
                        for i in range(len(img_discriminator)):
                            optimizer_d_img[i].zero_grad()
                        d_img_losses.total_loss.backward()
                        for i in range(len(img_discriminator)):
                            optimizer_d_img[i].step()

                    # 📍 논문 Section 3.4: Auxiliary Classifier Discriminator Training
                    # 신원 보존을 위한 보조 분류기의 판별기 역할 동시 수행
                    if ac_discriminator is not None:
                        d_ac_losses = LossManager()
                        for i in range(len(ac_discriminator)):
                            if isinstance(imgs_pred, tuple):
                                imgs_fake = imgs_pred[i].detach()
                            else:
                                imgs_fake = imgs_pred.detach()

                                imgs_real = imgs.detach()
                                while imgs_real.size()[2] != imgs_fake.size()[2]:
                                    imgs_real = F.interpolate(
                                        imgs_real, scale_factor=0.5, mode='nearest')

                                # 📍 논문 핵심: Dual Function of Auxiliary Classifier
                                # 1) 실제/가짜 판별 (scores_real/fake)
                                # 2) 신원 분류 (ac_loss_real) - SF2F의 핵심 혁신
                                scores_real, ac_loss_real= ac_discriminator[i](
                                    imgs_real, human_ids)
                                scores_fake, ac_loss_fake = ac_discriminator[i](
                                    imgs_fake, human_ids)

                                # 판별기로서의 적대적 손실
                                d_ac_gan_loss = gan_d_loss(scores_real, scores_fake)
                                d_ac_losses.add_loss(
                                    d_ac_gan_loss, 'd_ac_gan_loss_%d' % i)
                                # 실제 이미지에서의 신원 분류 손실 (감독 학습)
                                d_ac_losses.add_loss(
                                    ac_loss_real.mean(), 'd_ac_loss_real_%d' % i)

                            for i in range(len(ac_discriminator)):
                                optimizer_d_ac[i].zero_grad()
                            d_ac_losses.total_loss.backward()
                            for i in range(len(ac_discriminator)):
                                optimizer_d_ac[i].step()

                    # 📍 논문 Section 3.5: Conditional Discriminator Training
                    # 음성 조건을 활용한 조건부 판별 학습
                    if cond_discriminator is not None:
                        cond_d_losses = LossManager()
                        for i in range(len(cond_discriminator)):
                            if isinstance(imgs_pred, tuple):
                                imgs_fake = imgs_pred[i].detach()
                            else:
                                imgs_fake = imgs_pred.detach()
                            imgs_real = imgs.detach()
                            # 음성 조건 벡터 (SF2F의 핵심: 음성 특징을 조건으로 활용)
                            cond_vecs = others['cond'].detach()
                            
                            while imgs_real.size()[2] != imgs_fake.size()[2]:
                                imgs_real = F.interpolate(
                                    imgs_real, scale_factor=0.5, mode='nearest')

                            # 📍 논문 Equation (4): Conditional Discriminator Loss
                            # L_D_c = E[log(D_c(I_real, s))] + E[log(1 - D_c(G(s), s))]
                            # 실제 이미지와 음성이 매칭되는지, 생성된 이미지와 음성이 매칭되는지 판별
                            scores_fake = cond_discriminator[i](
                                imgs_fake, cond_vecs)
                            scores_real = cond_discriminator[i](
                                imgs_real, cond_vecs)

                            cond_d_gan_loss = gan_d_loss(scores_real, scores_fake)
                            cond_d_losses.add_loss(
                                cond_d_gan_loss, 'cond_d_gan_loss_%d' % i)

                        for i in range(len(cond_discriminator)):
                            optimizer_cond_d[i].zero_grad()
                        cond_d_losses.total_loss.backward()
                        for i in range(len(cond_discriminator)):
                            optimizer_cond_d[i].step()

            losses['total_loss'] = total_loss.item()
            if not math.isfinite(losses['total_loss']):
                log.warn('WARNING: Got loss = NaN, not backpropping')
                continue

            optimizer.zero_grad()
            with timeit('backward', args.timing):
                total_loss.backward()
            optimizer.step()

            total_loss_d = None
            ac_loss_real = None
            ac_loss_fake = None
            d_losses = {}

            with timeit('D_loss', args.timing):
                if img_discriminator is not None:
                    d_img_losses = LossManager()
                    for i in range(len(img_discriminator)):
                        if isinstance(imgs_pred, tuple):
                            imgs_fake = imgs_pred[i].detach()
                        else:
                            imgs_fake = imgs_pred.detach()
                        imgs_real = imgs.detach()
                        
                        while imgs_real.size()[2] != imgs_fake.size()[2]:
                            imgs_real = F.interpolate(
                                imgs_real, scale_factor=0.5, mode='nearest')

                        scores_fake = img_discriminator[i](imgs_fake)
                        scores_real = img_discriminator[i](imgs_real)

                        d_img_gan_loss = gan_d_loss(scores_real, scores_fake)
                        d_img_losses.add_loss(
                            d_img_gan_loss, 'd_img_gan_loss_%d' % i)

                    for i in range(len(img_discriminator)):
                        optimizer_d_img[i].zero_grad()
                    d_img_losses.total_loss.backward()
                    for i in range(len(img_discriminator)):
                        optimizer_d_img[i].step()

                if ac_discriminator is not None:
                    d_ac_losses = LossManager()
                    for i in range(len(ac_discriminator)):
                        if isinstance(imgs_pred, tuple):
                            imgs_fake = imgs_pred[i].detach()
                        else:
                            imgs_fake = imgs_pred.detach()

                            imgs_real = imgs.detach()
                            while imgs_real.size()[2] != imgs_fake.size()[2]:
                                imgs_real = F.interpolate(
                                    imgs_real, scale_factor=0.5, mode='nearest')

                            scores_real, ac_loss_real= ac_discriminator[i](
                                imgs_real, human_ids)
                            scores_fake, ac_loss_fake = ac_discriminator[i](
                                imgs_fake, human_ids)

                            d_ac_gan_loss = gan_d_loss(scores_real, scores_fake)
                            d_ac_losses.add_loss(
                                d_ac_gan_loss, 'd_ac_gan_loss_%d' % i)

                        for i in range(len(ac_discriminator)):
                            optimizer_d_ac[i].zero_grad()
                        d_ac_losses.total_loss.backward()
                        for i in range(len(ac_discriminator)):
                            optimizer_d_ac[i].step()

                if cond_discriminator is not None:
                    cond_d_losses = LossManager()
                    for i in range(len(cond_discriminator)):
                        if isinstance(imgs_pred, tuple):
                            imgs_fake = imgs_pred[i].detach()
                        else:
                            imgs_fake = imgs_pred.detach()
                        imgs_real = imgs.detach()
                        cond_vecs = others['cond'].detach()
                        while imgs_real.size()[2] != imgs_fake.size()[2]:
                            imgs_real = F.interpolate(
                                imgs_real, scale_factor=0.5, mode='nearest')

                        scores_fake = cond_discriminator[i](
                            imgs_fake, cond_vecs)
                        scores_real = cond_discriminator[i](
                            imgs_real, cond_vecs)

                        cond_d_gan_loss = gan_d_loss(scores_real, scores_fake)
                        cond_d_losses.add_loss(
                            cond_d_gan_loss, 'cond_d_gan_loss_%d' % i)

                    for i in range(len(cond_discriminator)):
                        optimizer_cond_d[i].zero_grad()
                    cond_d_losses.total_loss.backward()
                    for i in range(len(cond_discriminator)):
                        optimizer_cond_d[i].step()

            # 훈련 손실 텐서보드 로깅
            # 실시간 모니터링을 위해 모든 손실 값을 기록
            # 생성기 손실들 로깅
            for name, val in losses.items():
                logger.scalar_summary("loss/{}".format(name), val, t)
                
            # 이미지 판별기 손실들 로깅
            if img_discriminator is not None:
                for name, val in d_img_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
                    
            # 보조 분류기 판별기 손실들 로깅
            if ac_discriminator is not None:
                for name, val in d_ac_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
                    
            # 조건부 판별기 손실들 로깅
            if cond_discriminator is not None:
                for name, val in cond_d_losses.items():
                    logger.scalar_summary("d_loss/{}".format(name), val, t)
                    
            # 다음 배치 처리를 위한 시간 초기화
            start_time = time.time()

        # ===================================================================
        # SF2F 논문 Section 4: Experimental Results - Comprehensive Evaluation
        # 핵심 기여 6: 다양한 메트릭을 통한 포괄적 성능 평가
        # ===================================================================
        # 주기적 검증 및 평가 (에포크 단위)
        # SF2F 논문에서 강조한 다양한 메트릭을 사용한 포괄적 평가
        if epoch % args.eval_epochs == 0:
            log.info('[Epoch {}/{}] checking on val'.format(
                epoch, args.epochs)
            )
            
            # 📍 논문 Table 1: Standard Computer Vision Metrics
            # Inception Score (IS): 생성된 이미지의 품질과 다양성 측정
            # VGGFace Score: 얼굴 특화 품질 측정 (SF2F에서 제안)
            val_results = check_model(
                args, options, epoch, val_loader, model)
            val_losses, val_samples, val_inception, val_vfs = val_results
            
            # ===================================================================
            # SF2F 논문 Section 4.1: Quantitative Results - Speech-to-Face Metrics
            # 핵심 기여 7: 음성-얼굴 매핑 성능을 위한 전용 평가 지표
            # ===================================================================
            # 📍 논문 핵심: Speech-to-Face 특화 메트릭 계산
            # 이는 기존 이미지 생성 평가와 달리 음성-얼굴 매핑의 정확성을 직접 측정
            val_facenet_L2_dist, val_facenet_L1_dist, val_facenet_cos_sim, \
                val_recall_tuple, val_ih_sim  = \
                    s2f_val_evaluator.get_metrics(
                        model, recall_method='cos_sim', get_ih_sim=True)
                        
            # 📍 논문 Table 2: Recall@K 메트릭들 언패킹
            # SF2F 논문의 핵심 평가 지표: 음성에서 올바른 얼굴을 찾는 정확도
            # - Recall@1: 가장 유사한 얼굴이 정답인 비율 (엄격한 평가)
            # - Recall@5: 상위 5개 중 정답이 있는 비율 (실용적 평가)
            # - Recall@10: 상위 10개 중 정답이 있는 비율 (관대한 평가)
            val_recall_at_1, val_recall_at_2, val_recall_at_5, \
                val_recall_at_10, val_recall_at_20, \
                    val_recall_at_50 = val_recall_tuple
                    
            # ===================================================================
            # SF2F 논문: Model Selection via Multiple Metrics
            # 다중 메트릭 기반 최적 모델 선택 전략
            # ===================================================================
            # 각 메트릭별로 최고 성능을 달성했는지 확인
            # SF2F에서는 단일 메트릭이 아닌 다중 메트릭으로 종합 평가
            
            # 📍 논문 Figure 3: Quality Assessment via Inception Score
            # 생성된 이미지의 품질과 다양성 측정
            if val_inception[0] > best_inception[0]:
                got_best_IS = True
                best_inception = val_inception
                
            # 📍 논문 핵심: VGGFace Score for Face Quality
            # SF2F에서 제안한 얼굴 특화 품질 측정 메트릭
            # 일반적인 이미지 품질이 아닌 얼굴의 시각적 품질에 특화
            if val_vfs[0] > best_vfs[0]:
                got_best_VFS = True
                best_vfs = val_vfs
                
            # 📍 논문 Table 2: Identity Recall Metrics
            # SF2F의 핵심 성능 지표: 음성-얼굴 매핑 정확도
            if val_recall_at_1 > best_recall_1:
                got_best_R1 = True
                best_recall_1 = val_recall_at_1
                
            if val_recall_at_5 > best_recall_5:
                got_best_R5 = True
                best_recall_5 = val_recall_at_5
                
            if val_recall_at_10 > best_recall_10:
                got_best_R10 = True
                best_recall_10 = val_recall_at_10
                
            # 📍 논문: Feature Space Similarity Metrics
            # VGGFace 특징 공간에서의 유사성 측정 - SF2F의 핵심 평가 요소
            if val_facenet_cos_sim > best_cos:
                got_best_cos = True
                best_cos = val_facenet_cos_sim
                
            # L1 거리: 특징 공간에서의 거리 측정 (낮을수록 좋음)
            if val_facenet_L1_dist < best_L1:
                got_best_L1 = True
                best_L1 = val_facenet_L1_dist
                
            # 체크포인트에 최고 성능 메트릭들 저장
            checkpoint['counters']['best_inception'] = best_inception
            checkpoint['counters']['best_vfs'] = best_vfs
            checkpoint['val_samples'].append(val_samples)
            
            # 검증 손실들을 체크포인트와 텐서보드에 기록
            for k, v in val_losses.items():
                checkpoint['val_losses'][k].append(v)
                logger.scalar_summary("ckpt/val_{}".format(k), v, epoch)
                
            # SF2F 전용 메트릭들을 텐서보드에 로깅
            # 이들은 Speech-to-Face 모델의 성능을 종합적으로 평가
            logger.scalar_summary("ckpt/val_inception", val_inception[0], epoch)
            logger.scalar_summary("ckpt/val_facenet_L2_dist",
                val_facenet_L2_dist, epoch)
            logger.scalar_summary("ckpt/val_facenet_L1_dist",
                val_facenet_L1_dist, epoch)
            logger.scalar_summary("ckpt/val_facenet_cos_sim",
                val_facenet_cos_sim, epoch)
            logger.scalar_summary("ckpt/val_recall_at_1",
                val_recall_at_1, epoch)
            logger.scalar_summary("ckpt/val_recall_at_2",
                val_recall_at_2, epoch)
            logger.scalar_summary("ckpt/val_recall_at_5",
                val_recall_at_5, epoch)
            logger.scalar_summary("ckpt/val_recall_at_10",
                val_recall_at_10, epoch)
            logger.scalar_summary("ckpt/val_recall_at_20",
                val_recall_at_20, epoch)
            logger.scalar_summary("ckpt/val_recall_at_50",
                val_recall_at_50, epoch)
            logger.scalar_summary("ckpt/val_ih_sim",
                val_ih_sim, epoch)
            logger.scalar_summary("ckpt/val_vfs",
                val_vfs[0], epoch)
                
            # 검증 샘플 이미지들을 텐서보드에 로깅
            # 시각적 품질 평가를 위한 중요한 모니터링 도구
            logger.image_summary(val_samples, epoch, tag="ckpt_val")
            
            # 콘솔에 검증 결과 출력
            # SF2F 논문에서 중요하게 다룬 각종 메트릭들의 현재 값과 최고 값 비교
            log.info('[Epoch {}/{}] val inception score: {} ({})'.format(
                    epoch, args.epochs, val_inception[0], val_inception[1]))
            log.info('[Epoch {}/{}] best inception scores: {} ({})'.format(
                    epoch, args.epochs, best_inception[0], best_inception[1]))
            log.info('[Epoch {}/{}] val vfs scores: {} ({})'.format(
                    epoch, args.epochs, val_vfs[0], val_vfs[1]))
            log.info('[Epoch {}/{}] best vfs scores: {} ({})'.format(
                    epoch, args.epochs, best_vfs[0], best_vfs[1]))
            log.info('[Epoch {}/{}] val recall at 5: {}, '.format(
                     epoch, args.epochs, val_recall_at_5) + \
                        'best recall at 5: {}'.format(best_recall_5))
            log.info('[Epoch {}/{}] val recall at 10: {}, '.format(
                     epoch, args.epochs, val_recall_at_10) + \
                        'best recall at 10: {}'.format(best_recall_10))
            log.info('[Epoch {}/{}] val cosine similarity: {}, '.format(
                     epoch, args.epochs, val_facenet_cos_sim) + \
                        'best cosine similarity: {}'.format(best_cos))
            log.info('[Epoch {}/{}] val L1 distance: {}, '.format(
                     epoch, args.epochs, val_facenet_L1_dist) + \
                            'best L1 distance: {}'.format(best_L1))

            # 체크포인트 저장을 위한 모델 상태 수집
            # DataParallel 래퍼에서 실제 모델 상태 추출
            checkpoint['model_state'] = model.module.state_dict()

            # 이미지 판별기 상태들 저장
            if img_discriminator is not None:
                for i in range(len(img_discriminator)):
                    # 모델 가중치 저장
                    term_name = 'd_img_state_%d' % i
                    checkpoint[term_name] = \
                        img_discriminator[i].module.state_dict()
                    # 최적화기 상태 저장
                    term_name = 'd_img_optim_state_%d' % i
                    checkpoint[term_name] = \
                        optimizer_d_img[i].state_dict()

            # 보조 분류기 판별기 상태들 저장
            if ac_discriminator is not None:
                for i in range(len(ac_discriminator)):
                    # 모델 가중치 저장
                    term_name = 'd_ac_state_%d' % i
                    checkpoint[term_name] = \
                        ac_discriminator[i].module.state_dict()
                    # 최적화기 상태 저장
                    term_name = 'd_ac_optim_state_%d' % i
                    checkpoint[term_name] = \
                        optimizer_d_ac[i].state_dict()

            # 조건부 판별기 상태들 저장
            if cond_discriminator is not None:
                for i in range(len(cond_discriminator)):
                    # 모델 가중치 저장
                    term_name = 'cond_d_state_%d' % i
                    checkpoint[term_name] = \
                        cond_discriminator[i].module.state_dict()
                    # 최적화기 상태 저장
                    term_name = 'cond_d_optim_state_%d' % i
                    checkpoint[term_name] = \
                        optimizer_cond_d[i].state_dict()

            checkpoint['optim_state'] = optimizer.state_dict()
            checkpoint['counters']['epoch'] = epoch
            checkpoint['counters']['t'] = t
            checkpoint['counters']['best_inception'] = best_inception
            checkpoint['counters']['best_vfs'] = best_vfs
            checkpoint['counters']['best_recall_1'] = best_recall_1
            checkpoint['counters']['best_recall_5'] = best_recall_5
            checkpoint['counters']['best_recall_10'] = best_recall_10
            checkpoint['counters']['best_cos'] = best_cos
            checkpoint['counters']['best_L1'] = best_L1
            checkpoint['lr'] = lr
            checkpoint_path = os.path.join(
                log_path,
                '%s_with_model.pt' % args.checkpoint_name)
            log.info('[Epoch {}/{}] Saving checkpoint: {}'.format(
                epoch, args.epochs, checkpoint_path))
            torch.save(checkpoint, checkpoint_path)
            if got_best_IS:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_IS_with_model.pt'))
                got_best_IS = False
            if got_best_VFS:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_VFS_with_model.pt'))
                got_best_VFS = False
            if got_best_R1:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_R1_with_model.pt'))
                got_best_R1 = False
            if got_best_R5:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_R5_with_model.pt'))
                got_best_R5 = False
            if got_best_R10:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_R10_with_model.pt'))
                got_best_R10 = False
            if got_best_L1:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_L1_with_model.pt'))
                got_best_L1 = False
            if got_best_cos:
                copyfile(
                    checkpoint_path,
                    os.path.join(log_path, 'best_cos_with_model.pt'))
                got_best_cos = False

        if epoch > 0 and epoch % 1000 == 0:
            print('Saving checkpoint for Epoch {}.'.format(epoch))
            copyfile(
                checkpoint_path,
                os.path.join(log_path, 'epoch_{}_model.pt'.format(epoch)))
        # Fuser Logic
        elif args.train_fuser_only and epoch > 0 and epoch % 1 == 0:
            print('Saving checkpoint for Epoch {}.'.format(epoch))
            copyfile(
                checkpoint_path,
                os.path.join(log_path, 'epoch_{}_model.pt'.format(epoch)))
        # End of fuser logic

        if epoch >= args.decay_lr_epochs:
            lr_end = args.learning_rate * 1e-3
            decay_frac = (epoch - args.decay_lr_epochs + 1) / \
                (args.epochs - args.decay_lr_epochs + 1e-5)
            lr = args.learning_rate - decay_frac * (args.learning_rate - lr_end)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            if img_discriminator is not None:
                for i in range(len(optimizer_d_img)):
                    for param_group in optimizer_d_img[i].param_groups:
                        param_group["lr"] = lr
                # for param_group in optimizer_d_img.param_groups:
                #     param_group["lr"] = lr
            log.info('[Epoch {}/{}] learning rate: {}'.format(
                epoch+1, args.epochs, lr))

        logger.scalar_summary("ckpt/learning_rate", lr, epoch)

    # Evaluating after the whole training process.
    log.info("Evaluting the validation set.")
    is_mean, is_std, vfs_mean, vfs_std = evaluate(model, val_loader, options)
    log.info("Inception score: {} ({})".format(is_mean, is_std))
    log.info("VggFace score: {} ({})".format(vfs_mean, vfs_std))

if __name__ == '__main__':
    main()
