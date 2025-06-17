'''
This script generates a train/test/val split json for VoxCeleb dataset

vox1_meta.csv could be download from VoxCeleb official website:
    https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv
'''


import os
import json
import pandas as pd

VOX_DIR = os.path.join('./data/VoxCeleb')
vox1_meta_csv = os.path.join(VOX_DIR, 'vox1', 'vox1_meta.csv')
vox2_meta_csv = os.path.join(VOX_DIR, 'vox2', 'full_vox2_meta.csv')
split_json = os.path.join(VOX_DIR, 'split.json')



def main():
    vox1_df = pd.read_csv(vox1_meta_csv, sep='\t')
    vox2_df = pd.read_csv(vox2_meta_csv, sep='\t')

    vox1_df.columns = vox1_df.columns.str.strip()
    vox2_df.columns = vox2_df.columns.str.strip()

    print(vox1_df.head())
    print(vox2_df.head())

    split_dict = {
        'vox1': {'train':[], 'val':[], 'test':[]},
        'vox2': {'train':[], 'val':[], 'test':[]}
        }

    # 실제 사용 가능한 데이터를 기반으로 분할하기
    # 1. 먼저 실제 존재하는 identity들을 확인
    vox1_mel_available = set()
    vox1_face_available = set()
    vox2_mel_available = set()  
    vox2_face_available = set()
    
    # vox1 데이터 확인
    vox1_mel_dir = os.path.join(VOX_DIR, 'vox1', 'mel_spectrograms')
    vox1_face_dir = os.path.join(VOX_DIR, 'vox1', 'masked_faces')
    if os.path.exists(vox1_mel_dir):
        vox1_mel_available = set(os.listdir(vox1_mel_dir))
    if os.path.exists(vox1_face_dir):
        vox1_face_available = set(os.listdir(vox1_face_dir))
    
    # vox2 데이터 확인
    vox2_mel_dir = os.path.join(VOX_DIR, 'vox2', 'mel_spectrograms')
    vox2_face_dir = os.path.join(VOX_DIR, 'vox2', 'masked_faces')
    if os.path.exists(vox2_mel_dir):
        vox2_mel_available = set(os.listdir(vox2_mel_dir))
    if os.path.exists(vox2_face_dir):
        vox2_face_available = set(os.listdir(vox2_face_dir))
    
    # 교집합: 음성과 얼굴이 모두 있는 identity들
    vox1_available = vox1_mel_available.intersection(vox1_face_available)
    vox2_available = vox2_mel_available.intersection(vox2_face_available)
    
    print(f"VoxCeleb1 - 실제 사용 가능한 identity 수: {len(vox1_available)}")
    print(f"VoxCeleb2 - 실제 사용 가능한 identity 수: {len(vox2_available)}")
    
    # VoxCeleb1 분할: 실제 사용 가능한 데이터만
    vox1_available_list = sorted(list(vox1_available))
    for i, name in enumerate(vox1_available_list):
        if i % 10 == 8:
            split_dict['vox1']['test'].append(name)
        elif i % 10 == 9:
            split_dict['vox1']['val'].append(name)
        else:
            split_dict['vox1']['train'].append(name)

    # VoxCeleb2 분할: 실제 사용 가능한 데이터만
    vox2_available_list = sorted(list(vox2_available))
    for i, name in enumerate(vox2_available_list):
        if i % 10 == 8:
            split_dict['vox2']['test'].append(name)
        elif i % 10 == 9:
            split_dict['vox2']['val'].append(name)
        else:
            split_dict['vox2']['train'].append(name)

    print(len(split_dict['vox1']['train']),
          len(split_dict['vox1']['val']),
          len(split_dict['vox1']['test']))
    print(len(split_dict['vox2']['train']),
          len(split_dict['vox2']['val']),
          len(split_dict['vox2']['test']))

    with open(split_json, 'w') as outfile:
        json.dump(split_dict, outfile)

    return 0


if __name__ == '__main__':
    main()
