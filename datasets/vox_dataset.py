"""
==============================================================================
SF2F VoxCeleb Dataset: Speech-to-Face Paired Data Processing
==============================================================================

Paper: "Speech2Face: Learning a Face from a Voice"
URL: https://arxiv.org/abs/2006.05888

�� ����� SF2F(Speech Fusion to Face) ������ ����ϴ� VoxCeleb �����ͼ���
�ٽ� ������ ó�� Ŭ������ �����մϴ�.

SF2F ���� ������ ó�� �ٽ� Ư¡ (Core Data Processing Features):
=============================================================================

? 1. Multi-Modal Data Pairing (�� Section 3�� �ٽ� �Է�)
   - Speech Features: �α� �� ����Ʈ�α׷� (40-dimensional log mel-spectrograms)
   - Face Images: 128x128 �� �̹��� (VGG Face Detection ũ��)
   - Identity Labels: �ſ� ���� �н��� ���� ����� ���� ID

? 2. Temporal Audio Segmentation (���� ���� ó�� ����)
   - Variable-Length Audio: �پ��� ������ ���� ���׸�Ʈ ó��
   - Sliding Window: �ϰ��� ������ �н� ������ ����
   - Random Sampling: ������ ������ ���� ������ ���׸�Ʈ ����

? 3. Face-Audio Synchronization (���� �ٽ� ���� ����)
   - Same Identity Guarantee: ���� ����� ������ �� ��Ī ����
   - Multiple Views: �� ����� �پ��� ��/���� �������� robustness ���
   - Data Augmentation: ���� �ø�, ���׸�Ʈ ����Ʈ ��

? 4. Quality Control & Error Handling (���� ������ �߰�)
   - Corrupted Data Detection: �ջ�� �̹���/���� ���� �ڵ� ����
   - Fallback Mechanisms: ���� �߻� �� ��ü ������ ����
   - Empty Directory Filtering: �� ���丮 �ڵ� ����

? 5. Flexible Data Modes (���� �پ��� ���� ���� ����)
   - Single Segment Mode: �⺻ ��ġ�� �н���
   - Multi-Segment Mode: Fuser �Ʒ� �� �򰡿�
   - All Faces Mode: ���� �ι��� ��� �� �м���

Technical Implementation:
=============================================================================
- Mel Spectrogram Processing: 40-dim log mel features with temporal dynamics
- Image Preprocessing: VGG face cropping, ImageNet normalization
- Batch Collation: Variable-length audio handling with padding/cropping
- Memory Efficiency: Lazy loading and efficient caching strategies

�� �����ͼ� Ŭ������ SF2F ���� ����-�� ���� �н��� ����
��� ������ ó�� �䱸������ �����մϴ�.
"""

# �⺻ ���̺귯����
import json  # JSON ������ ������ ���� ���� ó��
import os  # ���� �ý��� ����
import numpy as np  # ��ġ ���� (�� ����Ʈ�α׷�, �̹��� �迭 ó��)
import random  # ������ ���ø� (������ ����)
import pickle  # �� ����Ʈ�α׷� ����ȭ ������ �ε�
import PIL  # �̹��� ó�� ���̺귯��
from PIL import Image  # PIL �̹��� ��ü
import torch  # PyTorch �ټ� ó��
from torch.utils.data import Dataset, DataLoader  # PyTorch �����ͼ� �⺻ Ŭ������
from torch.utils.data.dataloader import default_collate  # �⺻ ��ġ ���� �Լ�
import torchvision.transforms as T  # �̹��� ��ȯ (��������, ����ȭ ��)

# Ŀ���� ��ƿ��Ƽ �Լ��� (�̹���/�� ����Ʈ�α׷� ����ó��)
try:
    from .utils import imagenet_preprocess, imagenet_deprocess_batch, \
        fast_imagenet_deprocess_batch, fast_mel_deprocess_batch
except:
    from utils import imagenet_preprocess, imagenet_deprocess_batch, \
        fast_imagenet_deprocess_batch, fast_mel_deprocess_batch

# ===================================================================
# SF2F ��: VoxCeleb �����ͼ� �⺻ ���
# ===================================================================
VOX_DIR = os.path.join('./data', 'VoxCeleb')

class VoxDataset(Dataset):
    """
    ===================================================================
    SF2F ��: VoxCeleb ����-�� �� �����ͼ� Ŭ����
    ===================================================================
    
    ? ���� �ٽ� ������ ó�� �ý��� ����
    
    VoxCeleb1�� VoxCeleb2�� �����Ͽ� SF2F �Ʒÿ� �ʿ��� 
    ����-�� �� �����͸� ȿ�������� �����մϴ�.
    
    �ٽ� ���:
    - Multi-modal data pairing: ���� �ι��� ������ �� ��Ī
    - Temporal segmentation: ���� ���� ������ �ϰ��� ó��
    - Quality control: �ջ�� ������ �ڵ� ���� �� ó��
    - Flexible modes: �پ��� �Ʒ�/�� ���� ����
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
        SF2F VoxCeleb �����ͼ� �ʱ�ȭ
        ===================================================================
        
        ? �� Section 4: Experimental Setup�� ������ ���� ����
        
        Args:
            data_dir: VoxCeleb ������ ��Ʈ ���丮
            image_size: ��� �̹��� ũ�� (SF2F������ (128, 128) ǥ��)
            face_type: �� Ÿ��
                - 'masked': ��� ����ũ ó���� �� (SF2F ǥ��)
                - 'original': ���� �̹���
            image_normalize_method: �̹��� ����ȭ ���
                - 'imagenet': ImageNet ǥ�� ����ȭ (SF2F ���)
                - 'standard': ǥ�� ����ȭ
            mel_normalize_method: �� ����Ʈ�α׷� ����ȭ
                - 'vox_mel': VoxCeleb Ưȭ ����ȭ (SF2F ǥ��)
            nframe_range: �� ����Ʈ�α׷� ������ �� ���� (�ð��� ����)
            split_set: ������ ���� ('train', 'val', 'test')
            split_json: ������ ���� ���� JSON ����
            return_mel_segments: ��Ƽ ���׸�Ʈ ��� Ȱ��ȭ
                - False: ���� ���׸�Ʈ (�Ϲ� �Ʒ�)
                - True: ���� ���׸�Ʈ (Fuser �Ʒ�, ��)
            mel_seg_window_stride: �� ���׸�Ʈ ������ (ũ��, ��Ʈ���̵�)
            image_left_right_avg: �¿� �ø� ���ȭ (������ ����)
            image_random_hflip: ������ ���� �ø� (�Ʒ� �� ����)
            
        ? SF2F ���� ������ ó�� ����:
        - Multi-resolution support: �پ��� �̹��� �ػ� ����
        - Temporal dynamics: ������ �ð��� ��ȭ ����
        - Identity preservation: ����� �ϰ��� ���� �н�
        """
        # ===================================================================
        # SF2F ��: �⺻ ������ ����
        # ===================================================================
        self.data_dir = data_dir  # VoxCeleb ������ ��Ʈ ���
        self.image_size = image_size  # ��� �̹��� �ػ� (SF2F: 128x128)
        self.face_type = face_type  # �� Ÿ�� (SF2F: 'masked' ��ȣ)
        self.face_dir = self.face_type + '_faces'  # �� �̹��� ���丮��
        
        # ===================================================================
        # SF2F ��: ����ȭ �� ��ó�� ����
        # ===================================================================
        self.image_normalize_method = image_normalize_method  # �̹��� ����ȭ
        self.mel_normalize_method = mel_normalize_method  # �� ����ȭ
        
        # ===================================================================
        # SF2F ��: �ð��� ���� ó�� ����
        # ===================================================================
        self.nframe_range = nframe_range  # �� ����Ʈ�α׷� �ð��� ����
        # SF2F���� �ϰ��� �Է� ���̸� ���� ��� (��: 100-150 ������)
        
        # ===================================================================
        # SF2F ��: ������ ���� �� ��� ����
        # ===================================================================
        self.split_set = split_set  # ���� ����� ������ ����
        self.split_json = split_json  # ���� ���� ����
        
        # ===================================================================
        # SF2F ��: ��� ���� ó�� ���
        # ===================================================================
        self.return_mel_segments = return_mel_segments
        # ? False: ���� �� ���׸�Ʈ ��ȯ (�Ϲ� GAN �Ʒ�)
        # ? True: ���� �� ���׸�Ʈ ��ȯ (Fuser �Ʒ�, ��)
        
        self.mel_seg_window_stride = mel_seg_window_stride
        # ? �����̵� ������ ���� (������ ũ��, ��Ʈ���̵�)
        # SF2F���� �ð��� �ϰ��� �ִ� ���׸�Ʈ ������ ���
        
        self.shuffle_mel_segments = True
        # ? �� ���׸�Ʈ ���ø����� �ð��� ������ ����
        
        self.mel_segments_rand_start = False
        # ? ������ ���������� ������ �پ缺 ����
        # Fuser �����Ϳ��� Ư�� �߿� (�ð��� ��ȭ ����)
        
        # ===================================================================
        # SF2F ��: �̹��� ������ ���� ����
        # ===================================================================
        self.image_left_right_avg = image_left_right_avg
        # ? �¿� ��Ī ���ȭ: ���� ��Ī�� Ȱ���� ������ ����
        
        self.image_random_hflip = image_random_hflip
        # ? ������ ���� �ø�: �Ʒ� �� �Ϲ�ȭ ���� ���

        # ===================================================================
        # SF2F ��: �ʱ�ȭ ���μ��� ����
        # ===================================================================
        self.load_split_dict()  # ������ ���� ���� �ε�
        self.list_available_names()  # ��� ������ ������ ��� ����
        self.set_image_transform()  # �̹��� ��ȯ ���������� ����
        self.set_mel_transform()  # �� ����Ʈ�α׷� ��ȯ ���������� ����

    def __len__(self):
        """�����ͼ� ũ�� ��ȯ (��� ������ ��� ��)"""
        return len(self.available_names)

    def set_length(self, length):
        """�����ͼ� ũ�� ���� (�����, ���� �����)"""
        self.available_names = self.available_names[0:length]

    def __getitem__(self, index):
        """
        ===================================================================
        SF2F ��: ���� ������ �ε� �Լ�
        ===================================================================
        
        ? ���� �ٽ�: �־��� �ε������� ����-�� �� ��ȯ
        
        SF2F �Ʒ��� �⺻ ������ (�̹���, ����, �ſ�) ������ �����մϴ�.
        ������ ����� ������ ���� �������� �����Ͽ� ���� �н��� �����մϴ�.
        
        Args:
            index: ��� �ε��� (0 ~ len(dataset)-1)
            
        Returns:
            image: �� �̹��� �ټ� (C, H, W)
            log_mel: �α� �� ����Ʈ�α׷� �ټ� (F, T) �Ǵ� (N, F, T)
            human_id: ��� ID �ټ� (�ſ� ���� �н���)
            
        ? SF2F�� �ٽ� ������ ���Ἲ:
        - Same Identity: �ݵ�� ���� �ι��� ������ �� ��Ī
        - Quality Assurance: �ջ�� ���� �ڵ� ���� �� ó��
        - Random Sampling: ������ ������ ���� ������ ����
        """
        # ===================================================================
        # SF2F ��: �ſ� ���� ����
        # ===================================================================
        sub_dataset, name = self.available_names[index]
        # sub_dataset: 'vox1' �Ǵ� 'vox2'
        # name: ���� ����� ���� �ĺ��� (��: 'id10001')
        
        # ===================================================================
        # SF2F ��: �� �̹��� �ε� (Quality Control ����)
        # ===================================================================
        # ? �� �̹��� ���丮 ��� ����
        image_dir = os.path.join(
            self.data_dir, sub_dataset, self.face_dir, name)
        
        # ? �ջ�� �̹��� ó���� ���� Robust Loading
        image_files = os.listdir(image_dir)
        max_attempts = len(image_files)
        
        for attempt in range(max_attempts):
            try:
                # ? ������ �� �̹��� ���� (SF2F�� �پ缺 Ȯ�� ����)
                image_jpg = random.choice(image_files)
                image_path = os.path.join(image_dir, image_jpg)

                # ? ������ �̹��� �ε� �� ��ó��
                with open(image_path, 'rb') as f:
                    with PIL.Image.open(f) as image:
                        WW, HH = image.size
                        image = image.convert('RGB')  # RGB ���� ����
                        
                        # ? SF2F ������ ����: �¿� ��Ī ���ȭ
                        if self.image_left_right_avg:
                            # ������ ���� �ø��� ������� ��Ī�� ��ȭ
                            arr = (np.array(image) / 2.0 + \
                                np.array(T.functional.hflip(image)) / 2.0).astype(
                                    np.uint8)
                            image = PIL.Image.fromarray(arr, mode="RGB")
                            
                        # ? �̹��� ��ȯ ���������� ����
                        # (��������, ����ȭ, �ټ� ��ȯ ��)
                        image = self.image_transform(image)
                        break  # ���������� �ε�Ǹ� ���� ����
                        
            except (PIL.UnidentifiedImageError, OSError, IOError) as e:
                # ? �ջ�� �̹��� ���� ó��
                print(f"WARNING: Skipping corrupted image {image_path}: {e}")
                # �ջ�� ������ ��Ͽ��� �����Ͽ� �缱�� ����
                if image_jpg in image_files:
                    image_files.remove(image_jpg)
                if not image_files:  # ��� �̹����� �ջ�� ���
                    print(f"ERROR: All images corrupted for {name} in {sub_dataset}")
                    # ? Fallback: �⺻ ������ �̹��� ����
                    image = torch.zeros((3, self.image_size[0], self.image_size[1]))
                    break
                continue
        else:
            # ? ��� �õ� ���� �� �⺻ �̹��� ����
            print(f"ERROR: Could not load any image for {name} in {sub_dataset}")
            image = torch.zeros((3, self.image_size[0], self.image_size[1]))

        # ===================================================================
        # SF2F ��: �� ����Ʈ�α׷� �ε� (Quality Control ����)
        # ===================================================================
        # ? �� ����Ʈ�α׷� ���丮 ��� ����
        mel_gram_dir = os.path.join(
            self.data_dir, sub_dataset, 'mel_spectrograms', name)
        
        # ? ���丮 ���� �� ���� Ȯ�� (���� �ÿ��� ����� ���� ���)
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

        # ? �ջ�� �� ����Ʈ�α׷� ó���� ���� Robust Loading
        mel_files = os.listdir(mel_gram_dir)
        max_mel_attempts = len(mel_files)
        
        for attempt in range(max_mel_attempts):
            try:
                # ? ������ �� ����Ʈ�α׷� ����
                mel_gram_pickle = random.choice(mel_files)
                mel_gram_path = os.path.join(mel_gram_dir, mel_gram_pickle)
                
                if not self.return_mel_segments:
                    # ? ���� ���׸�Ʈ ��� (�Ϲ� GAN �Ʒ�)
                    log_mel = self.load_mel_gram(mel_gram_path)
                    log_mel = self.mel_transform(log_mel)
                    break
                    
            except (EOFError, pickle.UnpicklingError, OSError, IOError, KeyError) as e:
                # ? �ջ�� �� ���� ó��
                print(f"WARNING: Skipping corrupted mel file {mel_gram_path}: {e}")
                if mel_gram_pickle in mel_files:
                    mel_files.remove(mel_gram_pickle)
                if not mel_files:
                    print(f"ERROR: All mel files corrupted for {name} in {sub_dataset}")
                    # ? Fallback: �⺻ �� ����Ʈ�α׷� ����
                    log_mel = torch.zeros((40, 100))  # ǥ�� ũ�� (40-dim mel, 100 frames)
                    break
                continue
        else:
            if self.return_mel_segments:
                # ? ���� ���׸�Ʈ ��� (Fuser �Ʒ�, ��)
                # �� ��忡���� �� ����� ��� �� ���׸�Ʈ�� ��ȯ
                log_mel = self.get_all_mel_segments_of_id(
                    index, shuffle=self.shuffle_mel_segments)
            else:
                print(f"ERROR: Could not load any mel file for {name} in {sub_dataset}")
                log_mel = torch.zeros((40, 100))

        # ===================================================================
        # SF2F ��: �ſ� ���̺� ����
        # ===================================================================
        # ? �ſ� ���� �н��� ���� �ε��� ��� ID
        # ���� �з������ �� ID�� �ſ� �з��� �н�
        human_id = torch.tensor(index)

        return image, log_mel, human_id

    def get_all_faces_of_id(self, index):
        """
        ===================================================================
        SF2F ��: ���� �ι��� ��� �� �̹��� �ε�
        ===================================================================
        
        ? ���� �� ����: �� ����� ��� �󱼷� ��� �Ӻ��� ���
        
        �� �Լ��� SF2F�� ��� �� ��忡�� ���˴ϴ�:
        - ���� �� �̹����� FaceNet �Ӻ��� ���ȭ
        - ����, ���� ��ȭ�� robust�� ��ǥ �Ӻ��� ����
        - Recall@K ����� �� �������� ���� ����
        
        Args:
            index: ��� �ε���
            
        Returns:
            faces: ��� �� �̹��� ��ġ �ټ� (N, C, H, W)
        """
        sub_dataset, name = self.available_names[index]
        faces = []
        
        # ? �ش� �ι��� �� �̹��� ���丮
        image_dir = os.path.join(
            self.data_dir, sub_dataset, self.face_dir, name)
            
        # ? ��� �� �̹����� ���������� �ε�
        for image_jpg in os.listdir(image_dir):
            image_path = os.path.join(image_dir, image_jpg)
            with open(image_path, 'rb') as f:
                with PIL.Image.open(f) as image:
                    WW, HH = image.size
                    # ������ ��ó�� ���������� ����
                    image = self.image_transform(image.convert('RGB'))
                    faces.append(image)
                    
        # ? ��ġ �ټ��� ���� (N, C, H, W)
        faces = torch.stack(faces)
        return faces

    def get_all_mel_segments_of_id(self,
                                   index,
                                   shuffle=False):
        """
        ===================================================================
        SF2F ��: ���� �ι��� ��� ���� ���׸�Ʈ �ε�
        ===================================================================
        
        ? ���� �ٽ�: �ð��� �ϰ��� �ִ� ��Ƽ ���׸�Ʈ ó��
        
        �� �Լ��� SF2F�� Fuser ������Ʈ �Ʒð� �򰡿��� ���˴ϴ�:
        - �����̵� ������� �ϰ��� ������ ���׸�Ʈ ����
        - �ð��� ��ȭ�� �����ϴ� �پ��� ���� Ư¡ ����
        - ���� �Ӻ��� ���ȭ�� ���� �������� �� ����
        
        Args:
            index: ��� �ε���
            shuffle: �� ���� ���� ���ø� ����
            
        Returns:
            segments: ��� ���� ���׸�Ʈ ��ġ �ټ� (N, F, T)
        """
        sub_dataset, name = self.available_names[index]
        window_length, stride_length = self.mel_seg_window_stride
        segments = []
        
        # ? �ش� �ι��� �� ����Ʈ�α׷� ���丮
        mel_gram_dir = os.path.join(
            self.data_dir, sub_dataset, 'mel_spectrograms', name)
            
        # ? �� ���� ��� �غ�
        mel_gram_list = os.listdir(mel_gram_dir)
        if shuffle:
            random.shuffle(mel_gram_list)  # �ð��� ���� ������ȭ
        else:
            mel_gram_list.sort()  # �ϰ��� ���� ����
            
        seg_count = 0
        
        # ? �� �� ����Ʈ�α׷� ���Ͽ��� ���׸�Ʈ ����
        for mel_gram_pickle in mel_gram_list:
            mel_gram_path = os.path.join(mel_gram_dir, mel_gram_pickle)
            log_mel = self.load_mel_gram(mel_gram_path)
            log_mel = self.mel_transform(log_mel)
            mel_length = log_mel.shape[1]
            
            # ? ������ ������ ���� (������ �پ缺 ����)
            if self.mel_segments_rand_start:
                start = np.random.randint(mel_length - window_length) if mel_length > window_length else 0
                log_mel = log_mel[:, start:]
                mel_length = log_mel.shape[1]
                
            # ? �����̵� ������� ���׸�Ʈ ���� ���� �� ���
            num_window = 1 + (mel_length - window_length) // stride_length
            
            # ? �����̵� ������ ����
            for i in range(0, num_window):
                start_time = i * stride_length
                segment = log_mel[:, start_time:start_time + window_length]
                segments.append(segment)
                seg_count = seg_count + 1
                
                # ? ���׸�Ʈ �� ���� (�޸� ȿ����)
                if seg_count == 20:  # �ִ� 20�� ���׸�Ʈ
                    segments = torch.stack(segments)
                    return segments
                    
        segments = torch.stack(segments)
        return segments

    def set_image_transform(self):
        """
        ===================================================================
        SF2F ��: �̹��� ��ó�� ���������� ����
        ===================================================================
        
        ? ���� ǥ�� �̹��� ��ó�� ���� ����
        
        SF2F���� ����ϴ� �̹��� ��ó�� �ܰ�:
        1. ������ ���� �ø� (�Ʒ� �� ������ ����)
        2. ũ�� ���� (��ǥ �ػ󵵷� ��������)
        3. �ټ� ��ȯ (PIL �� PyTorch Tensor)
        4. ����ȭ (ImageNet ǥ�� �Ǵ� Ŀ����)
        """
        print('Dataloader: called set_image_size', self.image_size)
        
        # ? �⺻ ��ȯ ����������
        image_transform = [T.Resize(self.image_size), T.ToTensor()]
        
        # ? �Ʒ� �� ������ ���� �߰�
        if self.image_random_hflip and self.split_set == 'train':
            # 50% Ȯ���� ���� �ø� ���� (�� ��Ī�� Ȱ��)
            image_transform = [T.RandomHorizontalFlip(p=0.5),] + \
                image_transform
                
        # ? ����ȭ ����
        if self.image_normalize_method is not None:
            print('Dataloader: called image_normalize_method',
                self.image_normalize_method)
            # ImageNet ǥ�� ����ȭ �Ǵ� Ŀ���� ����ȭ
            image_transform.append(imagenet_preprocess(
                normalize_method=self.image_normalize_method))
                
        self.image_transform = T.Compose(image_transform)

    def set_mel_transform(self):
        """
        ===================================================================
        SF2F ��: �� ����Ʈ�α׷� ��ó�� ���������� ����
        ===================================================================
        
        ? ���� ǥ�� ���� Ư¡ ��ó�� ���� ����
        
        SF2F���� ����ϴ� �� ����Ʈ�α׷� ��ó�� �ܰ�:
        1. �ټ� ��ȯ (NumPy �� PyTorch Tensor)
        2. ����ȭ (VoxCeleb Ưȭ ����ȭ)
        3. ���� ���� (squeeze)
        """
        # ? �⺻ ��ȯ ����������
        mel_transform = [T.ToTensor(), ]
        
        print('Dataloader: called mel_normalize_method',
            self.mel_normalize_method)
            
        # ? �� ����Ʈ�α׷� ����ȭ ����
        if self.mel_normalize_method is not None:
            # VoxCeleb �����ͼ¿� Ưȭ�� ����ȭ ���
            mel_transform.append(imagenet_preprocess(
                normalize_method=self.mel_normalize_method))
                
        # ? ���ʿ��� ���� ����
        mel_transform.append(torch.squeeze)
        
        self.mel_transform = T.Compose(mel_transform)

    def load_split_dict(self):
        """
        ===================================================================
        SF2F ��: ������ ���� ���� �ε�
        ===================================================================
        
        ? ���� �� ��������: ����� �������� �Ϲ�ȭ ���� ����
        
        split.json���� ���� ������ �ε�:
        - train: �Ʒÿ� ��� ID ����Ʈ
        - val: ������ ��� ID ����Ʈ  
        - test: �׽�Ʈ�� ��� ID ����Ʈ
        
        �߿�: SF2F������ ��� ������ �����Ͽ� identity leakage ����
        """
        with open(self.split_json) as json_file:
            self.split_dict = json.load(json_file)

    def list_available_names(self):
        """
        ===================================================================
        SF2F ��: ��� ������ ������ ��� ����
        ===================================================================
        
        ? ���� ������ ���Ἲ ����: ������ �� ��� �����ϴ� ����� ����
        
        ����:
        1. VoxCeleb1�� VoxCeleb2���� ������ ��ĵ
        2. �� ����Ʈ�α׷��� �� �̹��� ������ ���
        3. ���� ����(train/val/test)�� ���ϴ� ����� ���͸�
        4. �� ���丮 ���� (ǰ�� ����)
        """
        self.available_names = []
        
        # ? VoxCeleb1�� VoxCeleb2 ���� ó��
        for sub_dataset in ('vox1', 'vox2'):
            # �� ����Ʈ�α׷� ��� ������ ��� ���
            mel_gram_available = os.listdir(
                os.path.join(self.data_dir, sub_dataset, 'mel_spectrograms'))
            # �� �̹��� ��� ������ ��� ���
            face_available = os.listdir(
                os.path.join(self.data_dir, sub_dataset, self.face_dir))
                
            # ? ������ ���: ������ �� ��� �ִ� ����� ����
            available = \
                set(mel_gram_available).intersection(face_available)
                
            for name in available:
                # ? ���� ���ҿ� ���ϴ��� Ȯ��
                if name in self.split_dict[sub_dataset][self.split_set]:
                    # ? �� ���丮 Ȯ�� (ǰ�� ����)
                    mel_dir = os.path.join(
                        self.data_dir, sub_dataset, 'mel_spectrograms', name)
                    face_dir = os.path.join(
                        self.data_dir, sub_dataset, self.face_dir, name)
                    
                    # ? ���丮 ���� �� ���� Ȯ��
                    if (os.path.exists(mel_dir) and len(os.listdir(mel_dir)) > 0 and
                        os.path.exists(face_dir) and len(os.listdir(face_dir)) > 0):
                        self.available_names.append((sub_dataset, name))
                    else:
                        print(f"WARNING: Skipping {name} in {sub_dataset} - empty directory")

        # ? �ϰ��� ���� ����
        self.available_names.sort()
        print(f"Total available samples after filtering: {len(self.available_names)}")

    def load_mel_gram(self, mel_pickle):
        """
        ===================================================================
        SF2F ��: �� ����Ʈ�α׷� �ε� �Լ�
        ===================================================================
        
        ? ���� ���� Ư¡: 40���� �α� �� ����Ʈ�α׷�
        
        Pickle ���Ͽ��� ���� ���� �ε�:
        - LogMel_Features: 40-dimensional log mel spectrogram
        - spkid: ȭ�� ID
        - clipid: Ŭ�� ID  
        - wavid: ���� ���� ID
        
        Args:
            mel_pickle: �� ����Ʈ�α׷� pickle ���� ���
            
        Returns:
            log_mel: �α� �� ����Ʈ�α׷� �迭 (F, T)
        """
        try:
            # ? Pickle ���Ͽ��� ������ �ε�
            with open(mel_pickle, 'rb') as file:
                data = pickle.load(file)
            log_mel = data['LogMel_Features']
            return log_mel
        except (EOFError, pickle.UnpicklingError, OSError, IOError, KeyError) as e:
            print(f"ERROR: Failed to load mel file {mel_pickle}: {e}")
            # ? Fallback: �⺻ �� ����Ʈ�α׷� ��ȯ
            # 40������ �Ϲ����� �� Ư¡ ������
            return np.zeros((40, 100), dtype=np.float32)

    def crop_or_pad(self, log_mel, out_frame):
        """
        ===================================================================
        SF2F ��: �� ����Ʈ�α׷� ���� ����ȭ
        ===================================================================
        
        ? ���� ��ġ ó��: ���� ���� ������ ���� ���̷� ����
        
        collate_fn�� �����Ͽ� ��ġ �� ��� �� ����Ʈ�α׷���
        ������ �ð� ���̷� ����ϴ�.
        
        Args:
            log_mel: �Է� �� ����Ʈ�α׷� �ټ�
            out_frame: ��ǥ ������ ��
            
        Returns:
            log_mel: ���̰� ������ �� ����Ʈ�α׷�
        """
        freq, cur_frame = log_mel.shape
        
        if cur_frame >= out_frame:
            # ? ũ��: ������ ��ġ���� ��ǥ ���̸�ŭ �ڸ���
            start = np.random.randint(0, cur_frame-out_frame+1)
            log_mel = log_mel[..., start:start+out_frame]
        else:
            # ? �е�: ������ ���̸� 0���� ä���
            zero_padding = np.zeros((freq, out_frame-cur_frame))
            zero_padding = self.mel_transform(zero_padding)
            if len(zero_padding.shape) == 1:
                zero_padding = zero_padding.view([-1, 1])
            log_mel = torch.cat([log_mel, zero_padding], -1)

        return log_mel

    def collate_fn(self, batch):
        """
        ===================================================================
        SF2F ��: Ŀ���� ��ġ ���� �Լ�
        ===================================================================
        
        ? ���� ��ġ ó��: ���� ���� ������ ���� ���� ����
        
        ��ġ �� ��� �� ����Ʈ�α׷��� �������� ���õ� 
        ������ ���̷� �����մϴ�. �̴� ������ �����մϴ�:
        - ȿ������ GPU ������ ���� ���� ũ�� �ټ�
        - �پ��� �ð� ���� �н����� �Ϲ�ȭ ���� ���
        - �޸� ��뷮 ����ȭ
        
        Args:
            batch: [(image, log_mel, human_id), ...] ������ ��ġ
            
        Returns:
            collated_batch: ���ϵ� ������ ��ġ
        """
        min_nframe, max_nframe = self.nframe_range
        assert min_nframe <= max_nframe
        
        # ? ��ġ���� ������ ������ �� ����
        np.random.seed()
        num_frame = np.random.randint(min_nframe, max_nframe+1)

        # ? �� ������ �� ����Ʈ�α׷��� ���õ� ���̷� ����
        batch = [(item[0],
                  self.crop_or_pad(item[1], num_frame),
                  item[2]) for item in batch]
                  
        # ? ǥ�� PyTorch ��ġ ���� ����
        return default_collate(batch)

    def count_faces(self):
        """
        ===================================================================
        �����ͼ� ���: ��ü �� �̹��� �� ���
        ===================================================================
        """
        total_count = 0
        for index in range(len(self.available_names)):
            sub_dataset, name = self.available_names[index]
            # �� �̹��� ���丮
            image_dir = os.path.join(
                self.data_dir, sub_dataset, self.face_dir, name)
            cur_count = len(os.listdir(image_dir))
            total_count = total_count + cur_count
        print('Number of faces in current dataset: {}'.format(total_count))
        return total_count

    def count_speech(self):
        """
        ===================================================================
        �����ͼ� ���: ��ü ���� ���� �� ���
        ===================================================================
        """
        total_count = 0
        for index in range(len(self.available_names)):
            sub_dataset, name = self.available_names[index]
            window_length, stride_length = self.mel_seg_window_stride
            # �� ����Ʈ�α׷� ���丮
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
    SF2F VoxDataset �׽�Ʈ �� ���� �ڵ�
    ===================================================================
    
    �� ������ �����ͼ� ������ ��Ȯ���� �����ϰ�
    �پ��� ��� ��ʸ� �׽�Ʈ�մϴ�.
    """
    
    # ? �� ������ ���ҿ� ���� �⺻ �׽�Ʈ
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

    # ? ��ġ ���� �Լ� �׽�Ʈ
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
