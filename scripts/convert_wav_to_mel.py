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

# �ڷ� ȯ��� �α� ���� ����
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ERROR�� ǥ��
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # GPU ����
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # oneDNN ����ȭ ��Ȱ��ȭ

# Python �α� ���� ����
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

# stderr �����̷������� CUDA �޽��� ����
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

# TensorFlow import �� ���� �޽��� ����
with SuppressStderr():
    from tensorflow.io import gfile

sys.path.append('./')

# wav2mel import �ÿ��� CUDA �޽��� ����
with SuppressStderr():
    from utils.wav2mel import wav_to_mel

from concurrent.futures import ProcessPoolExecutor, as_completed

# ��� ����
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
        # CSV ���� �ҷ����� (�� ������ ���)
        self.vox1_df = pd.read_csv(vox1_meta_csv, sep='\t')
        self.vox2_df = pd.read_csv(vox2_meta_csv, sep='\t')
        # �� �̸��� �յ� ���� ����
        self.vox1_df.columns = self.vox1_df.columns.str.strip()
        self.vox2_df.columns = self.vox2_df.columns.str.strip()

    def get_id2name(self):
        # ID�� �̸��� ����
        self.vox1_id2name = dict(
            zip(self.vox1_df['VoxCeleb1 ID'], self.vox1_df['VGGFace1 ID']))
        
        # Vox2 ���� ����
        self.vox2_id2name = {}
        for _, row in self.vox2_df.iterrows():
            vox2_id = row['VoxCeleb2 ID'].strip()
            name = row['Name'].strip()
            if vox2_id and name:  # �� ���� �ƴ� ��쿡�� �߰�
                self.vox2_id2name[vox2_id] = name
        
        # ����� ���� ���
        print(f"Vox1 ���� ��: {len(self.vox1_id2name)}")
        print(f"Vox2 ���� ��: {len(self.vox2_id2name)}")

    def find_empty_mel_directories(self):
        """
        �� mel spectrogram ���丮�� ã�Ƽ� �ش��ϴ� raw wav ���丮�� ��ȯ
        """
        empty_dirs = []
        
        for dataset in ['vox1', 'vox2']:
            mel_base_dir = os.path.join(VOX_DIR, dataset, 'mel_spectrograms')
            if not os.path.exists(mel_base_dir):
                print(f"? {mel_base_dir} ���丮�� �������� �ʽ��ϴ�.")
                continue
                
            print(f"? {dataset} mel_spectrograms ���丮 ��ĵ ��...")
            
            # mel_spectrograms ���丮�� ��� ���� ���丮 üũ
            for name in os.listdir(mel_base_dir):
                mel_dir_path = os.path.join(mel_base_dir, name)
                if os.path.isdir(mel_dir_path):
                    # ���丮�� ����ְų� ������ �ſ� ���� ���
                    files = os.listdir(mel_dir_path)
                    pickle_files = [f for f in files if f.endswith('.pickle')]
                    
                    if len(pickle_files) == 0:  # ������ �� ���丮
                        print(f"? �� ���丮 �߰�: {name} ({dataset})")
                        # �ش��ϴ� raw wav ���丮 ã��
                        corresponding_wav_dir = self.find_corresponding_wav_dir(name, dataset)
                        if corresponding_wav_dir:
                            empty_dirs.append((corresponding_wav_dir, getattr(self, f'{dataset}_mel'), dataset))
                    elif len(pickle_files) < 5:  # ������ �ʹ� ���� ���
                        print(f"? ���� ���� ���丮: {name} ({dataset}) - {len(pickle_files)}�� ����")
                        corresponding_wav_dir = self.find_corresponding_wav_dir(name, dataset)
                        if corresponding_wav_dir:
                            empty_dirs.append((corresponding_wav_dir, getattr(self, f'{dataset}_mel'), dataset))
        
        return empty_dirs

    def find_corresponding_wav_dir(self, name, dataset):
        """
        name(�̸�)�� �ش��ϴ� raw wav ���丮�� ã��
        """
        # �̸��� ID�� ����ȯ
        name_to_id = {}
        if dataset == 'vox1':
            name_to_id = {v: k for k, v in self.vox1_id2name.items()}
        elif dataset == 'vox2':
            name_to_id = {v: k for k, v in self.vox2_id2name.items()}
            
        speaker_id = name_to_id.get(name)
        if not speaker_id:
            print(f"?? {name}�� �ش��ϴ� ID�� ã�� �� �����ϴ�.")
            return None
            
        # raw wav ���丮���� �ش� ID ã��
        raw_base = getattr(self, f'{dataset}_raw')
        
        # dev, test ���丮���� ã��
        for split in ['dev', 'test']:
            split_dir = os.path.join(raw_base, split)
            if os.path.exists(split_dir):
                for dir_name in os.listdir(split_dir):
                    # ID�� ���Ե� ���丮 �̸� ã��
                    if dir_name.startswith(speaker_id):
                        wav_dir = os.path.join(split_dir, dir_name)
                        if os.path.isdir(wav_dir):
                            return wav_dir
        
        print(f"?? {speaker_id}({name})�� �ش��ϴ� raw wav ���丮�� ã�� �� �����ϴ�.")
        return None

    def convert_identity(self, wav_dir, mel_home_dir, dataset):
        spkid = wav_dir.split('/')[-1]
        if spkid == '':
            spkid = wav_dir.split('/')[-2]
        
        # _ ���� ID �κи� ���� (��: id01218_Brice_Hortefeux -> id01218)
        spkid = spkid.split('_')[0]
        spkid = spkid.strip()  # ���� ����

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
            print(f"  ��? clipid: {clipid} ó�� ����")
            wav_files = os.listdir(clip_dir)
            for wav_file in wav_files:
                wav_path = os.path.join(clip_dir, wav_file)
                try:
                    print(f"    ? wav ���� ó��: {wav_file}")
                    wavid = wav_file.replace('.wav', '').replace('.m4a', '')
                    pickle_name = f"{spkid}_{clipid}_{wavid}.pickle"
                    pickle_path = os.path.join(mel_dir, pickle_name)
                    if os.path.exists(pickle_path):
                        print(f"    ? �̹� ����: {pickle_name}, �ǳʶ�")
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
                    print(f"    ? ���� �Ϸ�: {pickle_name}")
                    processed_count += 1
                except Exception as e:
                    print(f"? Error processing {wav_path}: {e}")
        
        print(f"? {name}: �� {processed_count}�� ���� ó�� �Ϸ�")
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
        �� ���丮�� ã�Ƽ� ó��
        """
        print("? �� mel spectrogram ���丮 ��ĵ ��...")
        empty_dirs = self.find_empty_mel_directories()
        
        if not empty_dirs:
            print("? ��� ���丮�� mel spectrogram�� �ֽ��ϴ�!")
            return
            
        print(f"? ó���� �� ���丮: {len(empty_dirs)}��")
        for wav_dir, _, dataset in empty_dirs:
            print(f"  - {wav_dir} ({dataset})")
        
        # ��Ƽ���μ������� ó��
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

        print("? �� ���丮 ó�� �Ϸ�!")

    def convert_wav_to_mel(self, n_jobs=1):
        """
        ���� ���: ��� ���丮 ó��
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
                       help='�� ���丮�� ó���մϴ�')
    args = parser.parse_args()

    wav_convertor = WavConvertor()

    if args.empty_only:
        print("? �� ���丮�� ó�� ���")
        wav_convertor.convert_empty_directories_only(args.n_jobs)
    else:
        print("? ��ü ���丮 ó�� ���")
        wav_convertor.convert_wav_to_mel(args.n_jobs)

if __name__ == '__main__':
    main()