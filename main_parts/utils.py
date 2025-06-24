'''
utils.py

Functions that don't fit anywhere else.

'''
from PIL import Image
import numpy as np

ONE_BYTE_SCALE = 1.0 / 255.0


class EqMemorizedString:
    """ String that remembers what it was compared against """
    def __init__(self, string):
        self.string = string
        self.mem = set()

    def __eq__(self, other):
        self.mem.add(other)
        return self.string == other

    def mem_as_str(self):
        return ', '.join(self.mem)


'''
IMAGES
'''

def normalize_image(img_arr_uint):
    """
    Convert uint8 numpy image array into [0,1] float image array
    :param img_arr_uint:    [0,255]uint8 numpy image array
    :return:                [0,1] float32 numpy image array
    """
    return img_arr_uint.astype(np.float64) * ONE_BYTE_SCALE


def load_pil_image(filename, cfg):
    """Loads an image from a file path as a PIL image. Also handles resizing.

    Args:
        filename (string): path to the image file
        cfg (object): donkey configuration file

    Returns: a PIL image.
    """
    try:
        img = Image.open(filename)
        if img.height != cfg.IMAGE_H or img.width != cfg.IMAGE_W:
            img = img.resize((cfg.IMAGE_W, cfg.IMAGE_H))

        if cfg.IMAGE_DEPTH == 1:
            img = img.convert('L')
        
        return img

    except Exception as e:
        print(f'failed to load image from {filename}: {e.message}')
        return None


def load_image(filename, cfg):
    """
    :param string filename:     path to image file
    :param cfg:                 donkey config
    :return np.ndarray:         numpy uint8 image array
    """
    img_arr = load_image_sized(filename, cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)

    return img_arr


def load_image_sized(filename, image_width, image_height, image_depth):
    """Loads an image from a file path as a PIL image. Also handles resizing.

    Args:
        filename (string): path to the image file
        image_width: width in pixels of the output image
        image_height: height in pixels of the output image
        image_depth: depth of the output image (1 for greyscale)

    Returns:
        (np.ndarray):         numpy uint8 image array.
    """
    try:
        img = Image.open(filename)
        if img.height != image_height or img.width != image_width:
            img = img.resize((image_width, image_height))

        if image_depth == 1:
            img = img.convert('L')

        img_arr = np.asarray(img)

        # If the PIL image is greyscale, the np array will have shape (H, W)
        # Need to add a depth channel by expanding to (H, W, 1)
        if img.mode == 'L':
            h, w = img_arr.shape[:2]
            img_arr = img_arr.reshape(h, w, 1)

        return img_arr

    except Exception as e:
        print(f'failed to load image from {filename}: {e.message}')
        return None

def is_number_type(i):
    return type(i) == int or type(i) == float;

def get_model_by_type(model_type: str, cfg: 'Config'):
    '''
    given the string model_type and the configuration settings in cfg
    create a Keras model and return it.
    '''
    from parts.keras import KerasLinear, KerasIMU
    from parts.interpreter import TfLite, TensorRT

    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    if 'tflite_' in model_type:
        interpreter = TfLite()
        used_model_type = model_type.replace('tflite_', '')
    if 'tensorrt_' in model_type:
        interpreter = TensorRT()
        used_model_type = model_type.replace('tensorrt_', '')

    used_model_type = EqMemorizedString(used_model_type)
    if used_model_type == "linear":
        kl = KerasLinear(interpreter=interpreter, input_shape=input_shape)
    if used_model_type == "imu":
        kl = KerasIMU(interpreter=interpreter, input_shape=input_shape)

    return kl