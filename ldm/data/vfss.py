import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob

from vfss_data_split.datasets.vfss_dataset import VFSSImageDataset
import vfss_data_split.data_extraction.video_frame as video_frame

class VFSSDatasetBase(VFSSImageDataset):
    """VFSS Dataset Base for ldm.data"""
    def __init__(self, data_root, size=256, num_classes=2, **kwargs):
        self.video_frame_df = video_frame.load_video_frame_metadata_from_csv(data_root)
        super().__init__(video_frame_df=self.video_frame_df, **kwargs)
        self.size = size
        self.num_classes = num_classes
        
        self.data_paths = self.video_frame_df.image_path.tolist()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=self.video_frame_df.target_path.tolist())

        print(f"[Dataset]: VFSS with {self.video_frame_df.shape[0]} samples initialized.")

    def _validate_image_segmentation_size(self, image, segmentation):
        image_size = image.size
        segmentation_size = segmentation.size
        
        assert image_size == segmentation_size, "Image and segmentation size mismatch!"
        if self.size is not None:
            assert self.size > 0, "Size must be positive!"      
            assert image_size == (self.size, self.size), "Image size does not match the specified size!"
            assert segmentation_size == (self.size, self.size), "Segmentation size does not match the specified size!"

    def __getitem__(self, i):
        # Build example dict
        example = dict((k, self.labels[k][i]) for k in self.labels)
        
        # Read image and segmentation from VFSSImageDataset
        image, segmentation = super().__getitem__(i)
        image = transforms.ToPILImage()(image).convert("RGB")
        segmentation = transforms.ToPILImage()(segmentation).convert("RGB")

        # Validate sizes
        self._validate_image_segmentation_size(image, segmentation)
        
        # process segmentation
        segmentation = np.array(segmentation).astype(np.float32)
        if self.num_classes == 2:
            segmentation = (segmentation == 255).astype(np.float32)  # binary
        else:
            raise NotImplementedError("Only support binary segmentation now.")

        example["segmentation"] = (segmentation * 2) - 1  # range: binary -1 and 1

        # process image
        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.  # range from -1 to 1
        
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(example["segmentation"]) <= 1. and np.min(example["segmentation"]) >= -1.
        assert np.max(example["image"]) <= 1. and np.min(example["image"]) >= -1.
        return example

class VFSSTrain(VFSSDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(
            f'../dados_inca/metadados/video_frame_metadata_train.csv',
            target='mask',
            from_images=True,
            return_single_target=True,
            return_metadata=False,
            **kwargs
        )

class VFSSVal(VFSSDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(
            f'../dados_inca/metadados/video_frame_metadata_val.csv',
            target='mask',
            from_images=True,
            return_single_target=True, 
            return_metadata=False,
            **kwargs
        )
    
class VFSSTest(VFSSDatasetBase):
    def __init__(self, **kwargs):
        super().__init__(
            f'../dados_inca/metadados/video_frame_metadata_test.csv',
            target='mask',
            from_images=True,
            return_single_target=True,
            return_metadata=False,
            **kwargs
        )