import os

CAR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODELS_PATH = os.path.join(CAR_PATH, 'models')
DRIVE_LOOP_HZ = 20

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
