import copy
import os
from typing import Any, List, Optional, TypeVar, Iterator, Iterable
import numpy as np
from main_parts.config import Config
from parts.tub_v2 import Tub
from main_parts.utils import load_image, load_pil_image
from typing_extensions import TypedDict

TubRecordDict = TypedDict(
    'TubRecordDict',
    {
        '_index': int,
        'cam/image_array': str,
        'user/angle': float,
        'user/throttle': float,
        'user/mode': str,
        'imu/acl_x': Optional[float],
        'imu/acl_y': Optional[float],
        'imu/acl_z': Optional[float],
        'imu/gyr_x': Optional[float],
        'imu/gyr_y': Optional[float],
        'imu/gyr_z': Optional[float],
    }
)


class TubRecord(object):
    def __init__(self, config: Config, base_path: str,
                 underlying: TubRecordDict) -> None:
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._cache_images = getattr(self.config, 'CACHE_IMAGES', True)
        self._image: Optional[Any] = None

    def image(self, processor=None, as_nparray=True) -> np.ndarray:
        """
        Loads the image.

        :param processor:   Image processing like augmentations or cropping, if
                            not None. Defaults to None.
        :param as_nparray:  Whether to convert the image to a np array of uint8.
                            Defaults to True. If false, returns result of
                            Image.open()
        :return:            Image
        """
        if self._image is None:
            image_path = self.underlying['cam/image_array']
            full_path = os.path.join(self.base_path, 'images', image_path)

            if as_nparray:
                _image = load_image(full_path, cfg=self.config)
            else:
                # If you just want the raw Image
                _image = load_pil_image(full_path, cfg=self.config)
            if processor:
                _image = processor(_image)
            # only cache images if config does not forbid it
            if self._cache_images:
                self._image = _image
        else:
            _image = self._image
        return _image

    def __repr__(self) -> str:
        return repr(self.underlying)


class TubDataset(object):
    """
    Loads the dataset and creates a TubRecord list (or list of lists).
    """

    def __init__(self, config: Config, tub_paths: List[str],
                 seq_size: int = 0) -> None:
        self.config = config
        self.tub_paths = tub_paths
        self.tubs: List[Tub] = [Tub(tub_path, read_only=True)
                                for tub_path in self.tub_paths]
        self.records: List[TubRecord] = list()
        self.train_filter = getattr(config, 'TRAIN_FILTER', None)
        self.seq_size = seq_size

    def get_records(self):
        if not self.records:
            print(f'Loading tubs from paths {self.tub_paths}')
            for tub in self.tubs:
                for underlying in tub:
                    record = TubRecord(self.config, tub.base_path, underlying)
                    if not self.train_filter or self.train_filter(record):
                        self.records.append(record)
        return self.records
