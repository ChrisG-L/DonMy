import os
import numpy as np
from main_parts.config import Config
from parts.tub_v2 import Tub
from main_parts.utils import load_image, load_pil_image


class TubRecord(object):
    def __init__(self, config, base_path,
                 underlying):
        self.config = config
        self.base_path = base_path
        self.underlying = underlying
        self._image = None

    def image(self, processor=None, as_nparray=True):
        if self._image is None:
            image_path = self.underlying['cam/image_array']
            full_path = os.path.join(self.base_path, 'images', image_path)

            if as_nparray:
                _image = load_image(full_path)
            else:
                _image = load_pil_image(full_path)
            if processor:
                _image = processor(_image)
            self._image = _image
        else:
            _image = self._image
        return _image

    def __repr__(self):
        return repr(self.underlying)
