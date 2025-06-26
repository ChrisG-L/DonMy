import os
import time

import numpy as np
from PIL import Image

from parts.datastore_v2 import Manifest, ManifestIterator


class Tub(object):
    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_catalog_len=1000, read_only=False):
        self.base_path = base_path
        self.images_base_path = os.path.join(self.base_path, Tub.images())
        self.inputs = inputs
        self.types = types
        self.metadata = metadata
        self.manifest = Manifest(base_path, inputs=inputs, types=types,
                                 metadata=metadata, max_len=max_catalog_len,
                                 read_only=read_only)
        self.input_types = dict(zip(self.inputs, self.types))
        if not os.path.exists(self.images_base_path):
            os.makedirs(self.images_base_path, exist_ok=True)

    def write_record(self, record=None):
        contents = dict()
        for key, value in record.items():
            if value is None:
                continue
            elif key not in self.input_types:
                continue
            else:
                input_type = self.input_types[key]
                if input_type == 'float':
                    contents[key] = float(value)
                elif input_type == 'str':
                    contents[key] = value
                elif input_type == 'int':
                    contents[key] = int(value)
                elif input_type == 'boolean':
                    contents[key] = bool(value)
                elif input_type == 'nparray':
                    contents[key] = value.tolist()
                elif input_type == 'list' or input_type == 'vector':
                    contents[key] = list(value)
                elif input_type == 'image_array':
                    image = Image.fromarray(np.uint8(value))
                    name = Tub._image_file_name(self.manifest.current_index, key)
                    image_path = os.path.join(self.images_base_path, name)
                    image.save(image_path)
                    contents[key] = name
                elif input_type == 'gray16_array':
                    image = Image.fromarray(np.uint16(value))
                    name = Tub._image_file_name(self.manifest.current_index, key, ext='.png')
                    image_path = os.path.join(self.images_base_path, name)
                    image.save(image_path)
                    contents[key]=name

        contents['_timestamp_ms'] = int(round(time.time() * 1000))
        contents['_index'] = self.manifest.current_index
        contents['_session_id'] = self.manifest.session_id

        self.manifest.write_record(contents)

    def delete_last_n_records(self, n):
        all_alive_indexes = sorted(set(range(self.manifest.current_index))
                                   - self.manifest.deleted_indexes)
        to_delete_indexes = all_alive_indexes[-n:]
        self.manifest.delete_records(to_delete_indexes)

    def close(self):
        self.manifest.close()

    def __iter__(self):
        return ManifestIterator(self.manifest)

    def __len__(self):
        return self.manifest.__len__()

    @classmethod
    def images(cls):
        return 'images'

    @classmethod
    def _image_file_name(cls, index, key, extension='.jpg'):
        key_prefix = key.replace('/', '_')
        name = '_'.join([str(index), key_prefix, extension])
        return name


class TubWriter(object):
    def __init__(self, base_path, inputs=[], types=[], metadata=[],
                 max_catalog_len=1000):
        self.tub = Tub(base_path, inputs, types, metadata, max_catalog_len)

    def run(self, *args):
        assert len(self.tub.inputs) == len(args), \
            f'Expected {len(self.tub.inputs)} inputs but received {len(args)}'
        record = dict(zip(self.tub.inputs, args))
        self.tub.write_record(record)
        return self.tub.manifest.current_index

    def __iter__(self):
        return self.tub.__iter__()

    def close(self):
        self.tub.close()

    def shutdown(self):
        self.close()
