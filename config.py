import os

CAR_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
DRIVE_LOOP_HZ = 20

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3
CAMERA_FRAMERATE = DRIVE_LOOP_HZ

HAVE_IMU = False
IMU_SENSOR = 'mpu6050'
IMU_ADDRESS = 0x68
IMU_DLP_CONFIG = 0
