import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import einops

import torch
from torch.utils.data import Dataset 
import torchvision.transforms as T
from torchvision import transforms

from typing import Union, Optional

logger = logging.getLogger(__name__)

class VFSSWindowImageDataset(Dataset):
    def __init__(self, 
                 dataset_path: Union[str, Path], 
                 video_frame_table_filename='inca-video-frame-dataset.csv', 
                 window_size: int=1, 
                 stride: int = 1,  
                 size=256, 
                 return_metadata=True, 
                 image_transform=None, 
                 target_transform: list=None, 
                 repeat_channels=False,
                 image_interpolation="bilinear",
                 mask_interpolation="nearest",
                 split=None,
                 **kwargs
                ):
        '''
        Args:
            dataset_path (str | Path): Diretório raiz do dataset.
            video_frame_table_filename (str): Nome do arquivo CSV contendo o DataFrame com as informações dos frames de vídeo e seus alvos. 
            window_size (int): O número total de frames a serem carregados em cada janela (deve ser ímpar).
            stride (int): O passo entre os frames na janela.
            size (int): Dimensão de saída
            image_transform (callable, optional): Transformação a ser aplicada às imagens.
            target_transform (callable, optional): Transformação a ser aplicada à mascara.
            repeat_channels (bool): Se True, repete os canais das imagens em escala de cinza para criar imagens RGB.
            split (str, optional): O split do dataset a ser carregado (e.g., 'train', 'val', 'test'). Se None, carrega todo o dataset.
        '''

        self.dataset_path = Path(dataset_path)
        self.video_frame_df = pd.read_csv(self.dataset_path / video_frame_table_filename)
        self.video_frame_df['frame_id'] = self.video_frame_df['frame_id'].astype(int)
        self.video_frame_df['video_id'] = self.video_frame_df['video_id'].astype(int)
        assert self.video_frame_df.shape[0] > 0, "The video frame DataFrame is empty. Please check the provided dataset_path and video_frame_table_filename."
        
        if split is not None:
            self.split = split
            self.video_frame_df = self.video_frame_df[self.video_frame_df['split'] == split]
            self.video_frame_df.reset_index(drop=True, inplace=True)
        
        self.image_size = (size, size)
        self.return_metadata = return_metadata
        self.window_size = window_size
        self.stride = stride
        self.repeat_channels = repeat_channels
        self.image_interpolation = self.__parse_interpolation(image_interpolation)
        self.mask_interpolation = self.__parse_interpolation(mask_interpolation)
        
        if image_transform:
            self.image_transform = image_transform
        else:
            self.image_transform = T.Resize(
                self.image_size, interpolation=self.image_interpolation
            )
        
        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = T.Resize(
                self.image_size, interpolation=self.mask_interpolation
            )
    
    def __len__(self):
        return self.video_frame_df.shape[0]

    @staticmethod
    def __parse_interpolation(name):
        table = {
            "nearest": T.InterpolationMode.NEAREST,
            "bilinear": T.InterpolationMode.BILINEAR,
            "bicubic": T.InterpolationMode.BICUBIC,
            "lanczos": T.InterpolationMode.LANCZOS,
        }
        key = str(name).lower()
        if key not in table:
            raise ValueError(f"Unsupported interpolation: {name}")
        return table[key]

    def __load_mask_from_path(self, path: str):
        ''' Load mask image from the given path. '''
        
        path = self.__resolve_path(path)
        
        mask = Image.open(path).convert("L")
        mask = T.PILToTensor()(mask)

        return mask

    def __load_image(self, video_id: int, frame_id:int, path: str = None, color="greyscale"):
        '''
        Load image frame from path
        
        Args:
            video_id (int): The ID of the video to load the frame from.
            frame_id (int): The ID of the frame to load.
            path (str): The file path to load the image from.
            color (str): The color mode to load the image in ('greyscale' or 'rgb').

        Return:
            image (Tensor): The loaded image as a tensor.
        '''

        assert color in ['greyscale', 'rgb'], "Color mode must be either 'greyscale' or 'rgb'"
        assert path is not None or (
                video_id is not None 
                and frame_id is not None
            ), "Either path or both video_id and frame_id must be provided."
        
        # Check if path is provided, if not, resolve path using video_id and frame_id
        if path is None:
            video_df = self.video_frame_df[(self.video_frame_df.video_id == video_id)]
            video_frame_row = video_df[video_df.frame_id == frame_id]
            
            # If the specific frame is not found, we can use the path of labeled frame in the same video to construct the path for the desired frame.
            if video_frame_row.empty:
                labeled_frame_path = video_df.iloc[0].image_path
                image_folder = os.path.dirname(labeled_frame_path)
                path = os.path.join(image_folder, f"{frame_id}.png")
            else:
                path = video_frame_row.iloc[0].image_path

        path = self.__resolve_path(path)

        if color == 'greyscale':
            image_color = 'L'
        elif color == 'rgb':
            image_color = 'RGB'

        path = self.__resolve_path(path)
        image = Image.open(path).convert(image_color)
        image = T.PILToTensor()(image)

        return image
    
    def __resolve_path(self, path: str):
        path = self.dataset_path / path
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        return Path(path).expanduser().resolve()

    def get_total_frames_in_video(self, video_id: Union[str , int], filetype='.png'):
        ''' Get the total number of frames in a video based on the video_id '''
        video_frame_row = self.video_frame_df[self.video_frame_df.video_id == video_id].iloc[0]
        image_folder_path = os.path.dirname(video_frame_row.image_path)
        image_folder_path = self.__resolve_path(image_folder_path)

        frame_files = [f for f in os.listdir(image_folder_path) if f.endswith(filetype)]
        total_frames = len(frame_files)
        return total_frames

    def get_valid_window(self, frame_id, total_frames, window_size=3, stride=1, boundary_mode='repeat'):
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")
    
        half_window = window_size // 2
        valid_widow_frame_ids = {'begin': [],'end': []}
        for boundary in ['begin', 'end']:
            last_valid_frame_id = frame_id

            for i_window in range(1, half_window+1):
                # Determine the direction to check based on the boundary type. 
                #   For 'begin', we check frames before the current frame (negative direction)
                #   For 'end', we check frames after the current frame (positive direction).
                direction = -1 if boundary == 'begin' else 1
                check_frame_id = frame_id + direction * (i_window * stride)

                is_frame_id_valid = check_frame_id > 0 and check_frame_id <= total_frames
                if is_frame_id_valid:
                    valid_widow_frame_ids[boundary].append(check_frame_id)
                    last_valid_frame_id = check_frame_id
                else:
                    if boundary_mode == 'repeat':
                        valid_widow_frame_ids[boundary].append(last_valid_frame_id)
                    elif boundary_mode == 'constant':
                        # Use None to indicate out-of-bound frames
                        valid_widow_frame_ids[boundary].append(None)
                    else:
                        raise ValueError("Boundary mode must be either 'repeat' or 'constant'")

        window = valid_widow_frame_ids['begin'][::-1] + [frame_id] + valid_widow_frame_ids['end']
        return window

    def __load_window_images_from_path(self, video_id: Union[str, int], frame_id: Union[str, int], window_size: int, stride: int, color="greyscale", boundary_mode='repeat', repeat_channels=False):
        '''
        Load a window of image frames from a video based on the given video_id and frame_id.
        
        Args:
            video_id (str | int): The ID of the video to load frames from.
            frame_id (str | int): The ID of the central frame in the window to load.
            window_size (int): The total number of frames to load in the window (must be odd).
            stride (int): The stride between frames in the window.
            color (str): The color mode to load the images in ('greyscale' or 'rgb').
            boundary_mode (str): The mode to handle out-of-bound frames ('repeat' or 'constant').
                Handling out-of-bound frames:
                - 'repeat': Repeat the last valid frame when the window exceeds the video boundaries.
                - 'constant': Use a constant value (e.g., zero) for out-of-bound frames.

                Repeat mode example:
                For a video with 100 frames, window_size=5 and stride=1:
                - If frame_id=1 (beginning of the video)
                    Window frame IDs: [-2, -1, 0, 1, 2]
                    Resolved frame IDs with 'repeat': [1, 1, 1, 1, 2]
                - If frame_id=100 (end of the video)
                    Window frame IDs: [98, 99, 100, 101, 102]
                    Resolved frame IDs with 'repeat': [98, 99, 100, 100, 100]
                  
        Returns:
            images (Tensor): A tensor of size (window_size, C, H, W) containing the loaded image frames in the specified color mode.
        '''
        total_frames = self.get_total_frames_in_video(video_id)
        valid_window_frame_ids = self.get_valid_window(frame_id, total_frames, window_size, stride, boundary_mode)
        images = []
        for i, valid_frame_id in enumerate(valid_window_frame_ids):
            if valid_frame_id is not None:
                image = self.__load_image(video_id, valid_frame_id, color=color)
            else:
                n_channels = 3 if color == 'rgb' else 1
                image = torch.zeros((n_channels, self.image_size[0], self.image_size[1]))
            images.append(image)

        self.__current_image_original_dim = (images[0].shape[-1], images[0].shape[-2])
        images = torch.stack(images, dim=0)

        if repeat_channels and color == 'greyscale':
            images = images.repeat(1, 3, 1, 1)
        
        if window_size == 1:
            images = images.squeeze(0)

        return images, valid_window_frame_ids
    
    def __preprocess_image(self, image: torch.Tensor):
        '''Preprocess the image tensor (e.g., normalization)'''
        
        # logger.debug(f"Original image dimension: {image.shape} ({image.dtype}). Range: [{image.min()}, {image.max()}]")
        image = image.float()
        
        if image.max() > 1.0:
            image = image / 255.0

        image_min = image.amin(dim=[-1, -2], keepdim=True)
        image_max = image.amax(dim=[-1, -2], keepdim=True)
        
        # Normalize to [0, 1]
        image = (image - image_min) / (image_max - image_min + 1e-8)  
        image = (image * 2.0) - 1.0  # Scale to [-1, 1]

        if self.image_transform:
            image = self.image_transform(image)

        logger.debug(f"Preprocessed image dimension: {image.shape}. Range: [{image.min()}, {image.max()}]")
        return image
    
    def __preprocess_mask(self, mask: torch.Tensor):
        '''Preprocess the mask tensor (e.g., normalization)'''
        logger.debug(f"Original mask dimension: {mask.shape} ({mask.dtype}). Range: [{mask.min()}, {mask.max()}]")

        if mask.max() > 1.0:
            mask = mask.float() / 255.0

        if self.target_transform:
            mask = self.target_transform(mask)

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        # Binarize the mask
        mask = (mask > 0).float()  

        # Scale to [-1, 1]
        mask = (mask.float() * 2.0) - 1.0 
        
        if self.repeat_channels:
            mask = mask.unsqueeze(-1).repeat(1,1,3)
        
        logger.debug(f"Preprocessed mask dimension: {mask.shape} ({mask.dtype}). Range: [{mask.min()}, {mask.max()}]")
        return mask

    def __getitem__(self, idx=None, video_id=None, frame_id=None, video_frame=None):
        '''
        Get a sample from the dataset based on the provided identifiers.
        The sample includes a window of image frames and the corresponding ground truth mask, along with metadata

        Args:
            idx (int, optional): The index of the sample to retrieve. If provided, it takes precedence over video_id and frame_id.
            video_id (str | int, optional): The ID of the video to retrieve the sample from. Must be provided if idx is not provided.
            frame_id (str | int, optional): The ID of the frame to retrieve the sample from. Must be provided if idx is not provided.
            video_frame (str, optional): The combined identifier in the format "video_{video_id}_frame_{frame_id}". Must be provided if idx is not provided.
            split (str, optional): The dataset split to retrieve the sample from (e.g., 'train', 'val', 'test').
        
        Returns:
            A tuple containing:
            - image (Tensor): A tensor of size (window_size, C, H, W) containing the preprocessed image frames in the specified color mode.
            - gt_mask (Tensor): A tensor containing the preprocessed ground truth mask corresponding to the central frame in the window.
            - metadata (dict, optional): A dictionary containing metadata about the sample, including:
                - 'frame_id': The ID of the central frame in the window.
                - 'video_id': The ID of the video the sample belongs to.
                - 'video_frame': The combined identifier for the video and frame.
                - 'window_size': The size of the window used to load image frames.
                - 'stride': The stride used to load image frames in the window.
                - 'window_frame_ids': The list of frame IDs included in the loaded window of images.
                - 'original_dim': The original dimensions of the loaded images before preprocessing.
        '''

        if idx is None and (video_id is None or frame_id is None) and video_frame is None:
            raise ValueError("Either idx or (video_id and frame_id) or video_frame must be provided.")
        
        if idx is not None:
            row = self.video_frame_df[self.video_frame_df.index == idx]
        elif video_id is not None and frame_id is not None:
            row = self.video_frame_df[(self.video_frame_df.video_id == video_id) & (self.video_frame_df.frame_id == frame_id)]
        elif video_frame is not None:
            row = self.video_frame_df[self.video_frame_df.video_frame == video_frame]

        assert row is not None, "No matching row found for the given identifiers."
        assert row.shape[0] == 1, f"Multiple rows found for the given identifiers. Please ensure they uniquely identify a single row.\nidx={idx}\nrow={row}\nvideo_frame_df={self.video_frame_df}"
        
        row = row.iloc[0]

        frame_id = int(row.frame_id)
        video_id = int(row.video_id)
        frames, frame_ids = self.__load_window_images_from_path(
            video_id, 
            frame_id, 
            self.window_size, 
            self.stride, 
            color="greyscale", 
            boundary_mode='repeat',
            repeat_channels=self.repeat_channels
        )
        gt_mask = self.__load_mask_from_path(row.target_path)

        frames = self.__preprocess_image(frames)
        gt_mask = self.__preprocess_mask(gt_mask)

        metadata = {
            'frame_id': frame_id,
            'video_id': video_id,
            'video_frame': row.video_frame,
            'window_size': self.window_size,
            'stride': self.stride,
            'window_frame_ids': frame_ids,
            # 'paciente_id': row.paciente_id,
            # 'momento': row.momento,
            # 'procedimento': row.procedimento,
            'selected_labeler': row.selected_labeler,
            'original_dim': self.__current_image_original_dim
        }

        returns = {}
        returns['image'] = frames
        returns['segmentation'] = gt_mask
        returns['metadata'] = metadata

        # SDSeg Adaptations
        returns['class_id'] = np.array([-1]) # SDSeg requirement. Doesnt matter for binary segmentation.
        if returns['image'].ndim == 3:
            returns['image'] = einops.rearrange(returns['image'], 'c h w -> h w c')
        elif returns['image'].ndim != 4:
            raise ValueError(f"Unsupported VFSS image shape: {tuple(returns['image'].shape)}")
        returns['file_path_'] = row.target_path
        
        return returns

        
class VFSSIncaTrain(VFSSWindowImageDataset):
    def __init__(self, **kwargs):
        print(f"[Dataset]: Initializing VFSSIncaTrain with parameters: {kwargs}")
        super().__init__(
            split='train',
            **kwargs
        )


class VFSSIncaVal(VFSSWindowImageDataset):
    def __init__(self, **kwargs):
        super().__init__(
            split='val',
            **kwargs
        )
    
class VFSSIncaTest(VFSSWindowImageDataset):
    def __init__(self, **kwargs):
        super().__init__(
            split='test',
            **kwargs
        )
