import main_parts.config as cfg
from PIL import Image
import numpy as np

def normalize_image(img_arr_uint):
    return img_arr_uint.astype(np.float64) * (1.0 / 255.0)

def load_pil_image(filename):
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


def load_image(filename):
    img_arr = load_image_sized(filename, cfg.IMAGE_W, cfg.IMAGE_H, cfg.IMAGE_DEPTH)

    return img_arr


def load_image_sized(filename, image_width, image_height, image_depth):
    try:
        img = Image.open(filename)
        if img.height != image_height or img.width != image_width:
            img = img.resize((image_width, image_height))

        if image_depth == 1:
            img = img.convert('L')

        img_arr = np.asarray(img)

        if img.mode == 'L':
            h, w = img_arr.shape[:2]
            img_arr = img_arr.reshape(h, w, 1)

        return img_arr

    except Exception as e:
        print(f'failed to load image from {filename}: {e.message}')
        return None

def get_model(is_train):
    from parts.keras import KerasLinear
    from parts.interpreter import KerasInterpreter, TfLite

    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    if is_train:
        interpreter = KerasInterpreter()
    else:
        interpreter = TfLite()

    kl = KerasLinear(interpreter=interpreter, input_shape=input_shape)
    return kl
