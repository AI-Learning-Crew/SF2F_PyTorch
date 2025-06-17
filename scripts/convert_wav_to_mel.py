'''
Convert raw_wavs in VoxCeleb dataSet to mel_gram

    1. Build two id2name mapping for vox1 & vox2
    2. Create a folder for each identity with his/her name
    3. For each raw wav file, convert it to mel_gram,
       and save it under the identity folder
'''

import os
import logging
import sys

# 코랩 환경용 로깅 억제 설정
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR만 표시
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU 지정
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN 최적화 비활성화

# Python 로깅 레벨 설정
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# stderr 리다이렉션으로 CUDA 메시지 억제
class SuppressStderr:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

import warnings
warnings.filterwarnings("ignore")
import argparse
import shutil
import gc
import time
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp

# TensorFlow import 시 에러 메시지 억제
with SuppressStderr():
    from tensorflow.io import gfile

sys.path.append('./')

# wav2mel import 시에도 CUDA 메시지 억제
with SuppressStderr():
    from utils.wav2mel import wav_to_mel

from concurrent.futures import ProcessPoolExecutor, as_completed

# 경로 설정
VOX_DIR = os.path.join('./data/VoxCeleb')
vox1_raw = os.path.join(VOX_DIR, 'raw_wav', 'vox1')
vox2_raw = os.path.join(VOX_DIR, 'raw_wav', 'vox2')
vox1_meta_csv = os.path.join(VOX_DIR, 'vox1', 'vox1_meta.csv')
vox2_meta_csv = os.path.join(VOX_DIR, 'vox2', 'full_vox2_meta.csv')

class WavConvertor:
    def __init__(self):
        self.load_metadata()
        self.get_id2name()
        self.get_wav_dirs()
        self.create_output_dirs()

    def load_metadata(self):
        # CSV 파일 불러오기 (탭 구분자 사용)
        self.vox1_df = pd.read_csv(vox1_meta_csv, sep='\t')
        self.vox2_df = pd.read_csv(vox2_meta_csv, sep='\t')
        # 열 이름의 앞뒤 공백 제거
        self.vox1_df.columns = self.vox1_df.columns.str.strip()
        self.vox2_df.columns = self.vox2_df.columns.str.strip()

    def get_id2name(self):
        # ID와 이름을 매핑
        self.vox1_id2name = dict(
            zip(self.vox1_df['VoxCeleb1 ID'], self.vox1_df['VGGFace1 ID']))
        
        # Vox2 매핑 개선
        self.vox2_id2name = {}
        for _, row in self.vox2_df.iterrows():
            vox2_id = row['VoxCeleb2 ID'].strip()
            name = row['Name'].strip()
            if vox2_id and name:  # 빈 값이 아닌 경우에만 추가
                self.vox2_id2name[vox2_id] = name
        
        # 디버깅 정보 출력
        print(f"Vox1 매핑 수: {len(self.vox1_id2name)}")
        print(f"Vox2 매핑 수: {len(self.vox2_id2name)}")

    def find_empty_mel_directories(self):
        """
        빈 mel spectrogram 디렉토리를 찾아서 해당하는 raw wav 디렉토리를 반환
        """
        empty_dirs = []
        
        for dataset in ['vox1', 'vox2']:
            mel_base_dir = os.path.join(VOX_DIR, dataset, 'mel_spectrograms')
            if not os.path.exists(mel_base_dir):
                print(f"? {mel_base_dir} 디렉토리가 존재하지 않습니다.")
                continue
                
            print(f"? {dataset} mel_spectrograms 디렉토리 스캔 중...")
            
            # mel_spectrograms 디렉토리의 모든 하위 디렉토리 체크
            for name in os.listdir(mel_base_dir):
                mel_dir_path = os.path.join(mel_base_dir, name)
                if os.path.isdir(mel_dir_path):
                    # 디렉토리가 비어있거나 파일이 매우 적은 경우
                    files = os.listdir(mel_dir_path)
                    pickle_files = [f for f in files if f.endswith('.pickle')]
                    
                    if len(pickle_files) == 0:  # 완전히 빈 디렉토리
                        print(f"? 빈 디렉토리 발견: {name} ({dataset})")
                        # 해당하는 raw wav 디렉토리 찾기
                        corresponding_wav_dir = self.find_corresponding_wav_dir(name, dataset)
                        if corresponding_wav_dir:
                            empty_dirs.append((corresponding_wav_dir, getattr(self, f'{dataset}_mel'), dataset))
                    elif len(pickle_files) < 5:  # 파일이 너무 적은 경우
                        print(f"? 파일 부족 디렉토리: {name} ({dataset}) - {len(pickle_files)}개 파일")
                        corresponding_wav_dir = self.find_corresponding_wav_dir(name, dataset)
                        if corresponding_wav_dir:
                            empty_dirs.append((corresponding_wav_dir, getattr(self, f'{dataset}_mel'), dataset))
        
        return empty_dirs

    def find_corresponding_wav_dir(self, name, dataset):
        """
        name(이름)에 해당하는 raw wav 디렉토리를 찾기
        """
        # 이름을 ID로 역변환
        name_to_id = {}
        if dataset == 'vox1':
            name_to_id = {v: k for k, v in self.vox1_id2name.items()}
        elif dataset == 'vox2':
            name_to_id = {v: k for k, v in self.vox2_id2name.items()}
            
        speaker_id = name_to_id.get(name)
        if not speaker_id:
            print(f"?? {name}에 해당하는 ID를 찾을 수 없습니다.")
            return None
            
        # raw wav 디렉토리에서 해당 ID 찾기
        raw_base = getattr(self, f'{dataset}_raw')
        
        # dev, test 디렉토리에서 찾기
        for split in ['dev', 'test']:
            split_dir = os.path.join(raw_base, split)
            if os.path.exists(split_dir):
                for dir_name in os.listdir(split_dir):
                    # ID가 포함된 디렉토리 이름 찾기
                    if dir_name.startswith(speaker_id):
                        wav_dir = os.path.join(split_dir, dir_name)
                        if os.path.isdir(wav_dir):
                            return wav_dir
        
        print(f"?? {speaker_id}({name})에 해당하는 raw wav 디렉토리를 찾을 수 없습니다.")
        return None

    def convert_identity(self, wav_dir, mel_home_dir, dataset):
        spkid = wav_dir.split('/')[-1]
        if spkid == '':
            spkid = wav_dir.split('/')[-2]
        
        # _ 앞의 ID 부분만 추출 (예: id01218_Brice_Hortefeux -> id01218)
        spkid = spkid.split('_')[0]
        spkid = spkid.strip()  # 공백 제거

        name = None
        if dataset == 'vox1':
            name = self.vox1_id2name.get(spkid)
        elif dataset == 'vox2':
            name = self.vox2_id2name.get(spkid)
            if name is None:
                print(f"?? ID {spkid} not found in vox2_id2name mapping")
                return

        if not name:
            print(f"? ID {spkid} not found in metadata")
            return

        print(f"? Processing: {spkid} -> {name}")

        mel_dir = os.path.join(mel_home_dir, name)
        if not os.path.exists(mel_dir):
            gfile.mkdir(mel_dir)

        processed_count = 0
        clipids = os.listdir(wav_dir)
        for clipid in clipids:
            clip_dir = os.path.join(wav_dir, clipid)
            if not os.path.isdir(clip_dir):
                continue
            print(f"  ▶? clipid: {clipid} 처리 시작")
            wav_files = os.listdir(clip_dir)
            for wav_file in wav_files:
                wav_path = os.path.join(clip_dir, wav_file)
                try:
                    print(f"    ? wav 파일 처리: {wav_file}")
                    wavid = wav_file.replace('.wav', '').replace('.m4a', '')
                    pickle_name = f"{spkid}_{clipid}_{wavid}.pickle"
                    pickle_path = os.path.join(mel_dir, pickle_name)
                    if os.path.exists(pickle_path):
                        print(f"    ? 이미 존재: {pickle_name}, 건너뜀")
                        continue
                    log_mel = wav_to_mel(wav_path)
                    pickle_dict = {
                        'LogMel_Features': log_mel,
                        'spkid': spkid,
                        'clipid': clipid,
                        'wavid': wavid
                    }
                    with open(pickle_path, "wb") as f:
                        pickle.dump(pickle_dict, f)
                    print(f"    ? 저장 완료: {pickle_name}")
                    processed_count += 1
                except Exception as e:
                    print(f"? Error processing {wav_path}: {e}")
        
        print(f"? {name}: 총 {processed_count}개 파일 처리 완료")
        gc.collect()

    def get_wav_dirs(self):
        vox1_dev = os.path.join(vox1_raw, 'dev')
        vox1_test = os.path.join(vox1_raw, 'test')
        vox2_dev = os.path.join(vox2_raw, 'dev')
        vox2_test = os.path.join(vox2_raw, 'test')

        vox1_wav_dirs = [os.path.join(vox1_dev, d) for d in os.listdir(vox1_dev) if os.path.isdir(os.path.join(vox1_dev, d))]
        vox1_wav_dirs += [os.path.join(vox1_test, d) for d in os.listdir(vox1_test) if os.path.isdir(os.path.join(vox1_test, d))]
        vox2_wav_dirs = [os.path.join(vox2_dev, d) for d in os.listdir(vox2_dev) if os.path.isdir(os.path.join(vox2_dev, d))]
        vox2_wav_dirs += [os.path.join(vox2_test, d) for d in os.listdir(vox2_test) if os.path.isdir(os.path.join(vox2_test, d))]

        self.vox1_wav_dirs = vox1_wav_dirs
        self.vox2_wav_dirs = vox2_wav_dirs
        return 0

    def create_output_dirs(self):
        self.vox1_mel = os.path.join(VOX_DIR, 'vox1', 'mel_spectrograms')
        self.vox2_mel = os.path.join(VOX_DIR, 'vox2', 'mel_spectrograms')
        if not os.path.exists(self.vox1_mel):
            gfile.mkdir(self.vox1_mel)
        if not os.path.exists(self.vox2_mel):
            gfile.mkdir(self.vox2_mel)

    def _worker(self, job_id, infos):
        for i, info in enumerate(infos):
            self.convert_identity(info[0], info[1], info[2])
            print(f'[Job {job_id}] Processed {i+1}/{len(infos)}: {info[0]}')

    def convert_empty_directories_only(self, n_jobs=1):
        """
        빈 디렉토리만 찾아서 처리
        """
        print("? 빈 mel spectrogram 디렉토리 스캔 중...")
        empty_dirs = self.find_empty_mel_directories()
        
        if not empty_dirs:
            print("? 모든 디렉토리에 mel spectrogram이 있습니다!")
            return
            
        print(f"? 처리할 빈 디렉토리: {len(empty_dirs)}개")
        for wav_dir, _, dataset in empty_dirs:
            print(f"  - {wav_dir} ({dataset})")
        
        # 멀티프로세싱으로 처리
        n_dirs = len(empty_dirs)
        n_jobs = min(n_jobs, n_dirs)
        n_per_job = n_dirs // n_jobs

        process_index = [[i * n_per_job, (i + 1) * n_per_job] for i in range(n_jobs)]
        if n_jobs > 0:
            process_index[-1][1] = n_dirs

        futures = set()
        with ProcessPoolExecutor() as executor:
            for job_id, (start, end) in enumerate(process_index):
                future = executor.submit(self._worker, job_id, empty_dirs[start:end])
                futures.add(future)
                print(f"? Submitted job {job_id}: {start} to {end}")
            for future in as_completed(futures):
                pass

        print("? 빈 디렉토리 처리 완료!")

    def convert_wav_to_mel(self, n_jobs=1):
        """
        기존 방식: 모든 디렉토리 처리
        """
        infos = [(d, self.vox1_mel, 'vox1') for d in self.vox1_wav_dirs]
        infos += [(d, self.vox2_mel, 'vox2') for d in self.vox2_wav_dirs]

        n_wav_dirs = len(infos)
        n_jobs = min(n_jobs, n_wav_dirs)
        n_per_job = n_wav_dirs // n_jobs

        process_index = [[i * n_per_job, (i + 1) * n_per_job] for i in range(n_jobs)]
        if n_jobs > 0:
            process_index[-1][1] = n_wav_dirs

        futures = set()
        with ProcessPoolExecutor() as executor:
            for job_id, (start, end) in enumerate(process_index):
                future = executor.submit(self._worker, job_id, infos[start:end])
                futures.add(future)
                print(f"? Submitted job {job_id}: {start} to {end}")
            for future in as_completed(futures):
                pass

        print("? All jobs completed.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_jobs', '-n_jobs', type=int, default=1)
    parser.add_argument('--empty_only', action='store_true', 
                       help='빈 디렉토리만 처리합니다')
    args = parser.parse_args()

    wav_convertor = WavConvertor()

    if args.empty_only:
        print("? 빈 디렉토리만 처리 모드")
        wav_convertor.convert_empty_directories_only(args.n_jobs)
    else:
        print("? 전체 디렉토리 처리 모드")
        wav_convertor.convert_wav_to_mel(args.n_jobs)

if __name__ == '__main__':
    main()