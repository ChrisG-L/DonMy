import os
from typing import Any, List, Optional
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
        'user/mode': str
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
                _image = load_image(full_path)
            else:
                # If you just want the raw Image
                _image = load_pil_image(full_path)
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
