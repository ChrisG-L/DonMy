#!/usr/bin/env python3
#
import argparse
import time

from main_parts.vehicle import Vehicle
from main_parts.utils import get_model_by_type
from main_parts.config import load_config
from parts.classUtils import TriggeredCallback, DelayedTrigger, ToggleRecording, Pipe, DriveMode

from parts.oakd_camera import OakDCamera
from parts.tub_v2 import TubWriter

def drive(cfg, model_path="./models/mypilot.tflite", use_joystick=False, model_type="tflite_linear"):

    V = Vehicle()

    cam = OakDCamera(width=cfg.IMAGE_W,
            height=cfg.IMAGE_H,
            fps=cfg.CAMERA_FRAMERATE)
    V.add(cam, inputs=[], outputs=['cam/image_array'], threaded=True)

    ctr = add_user_controller(V, cfg, use_joystick)

    V.add(Pipe(), inputs=['user/steering'], outputs=['user/angle'])

    add_imu(V, cfg)

    def load_model(kl, model_path):
        start = time.time()
        print("\n\n\nLOAD\n\n\n")
        print('loading model', model_path)
        kl.load(model_path)
        print('finished loading in %s sec.' % (str(time.time() - start)) )

    if not use_joystick:
        kl = get_model_by_type(model_type, cfg)

        model_reload_cb = None
        load_model(kl, model_path)

        def reload_model(filename):
            print("\n\n\nModel Reload\n\n\n")
            load_model(kl, filename)

        model_reload_cb = reload_model

        V.add(DelayedTrigger(100), inputs=['modelfile/dirty'],
              outputs=['modelfile/reload'], run_condition="run_pilot")
        V.add(TriggeredCallback(model_path, model_reload_cb),
              inputs=["modelfile/reload"], run_condition="run_pilot")

        if model_type == "imu":
            assert cfg.HAVE_IMU, 'Missing imu parameter in config'

            class Vectorizer:
                def run(self, *components):
                    return components

            V.add(Vectorizer, inputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                                      'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'],
                  outputs=['imu_array'])

            inputs = ['cam/image_array', 'imu_array']
        else:
            inputs = ['cam/image_array']

        outputs = ['pilot/angle', 'pilot/throttle']

        V.add(kl, inputs=inputs, outputs=outputs, run_condition='run_pilot')

    V.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['steering', 'throttle'])

    recording_control = ToggleRecording()
    V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])

    add_drivetrain(V)

    inputs=['cam/image_array','user/angle', 'user/throttle', 'user/mode']
    types=['image_array','float', 'float','str']

    if cfg.HAVE_IMU:
        inputs += ['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z']

        types +=['float', 'float', 'float',
           'float', 'float', 'float']

    tub_path = cfg.DATA_PATH
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    if ctr is not None:
        print("You can now move your controller to drive your car.")
        ctr.set_tub(tub_writer.tub)

    V.start(rate_hz=cfg.DRIVE_LOOP_HZ)

def add_user_controller(V, cfg, use_joystick, input_image='ui/image_array'):
    ctr = None
    if use_joystick:
        from parts.controller import get_js_controller
        ctr = get_js_controller(cfg)
        V.add(
            ctr,
            inputs=[input_image, 'user/mode', 'recording'],
            outputs=['user/steering', 'user/throttle',
                        'user/mode', 'recording'],
            threaded=True)
    return ctr

def add_imu(V, cfg):
    imu = None
    if cfg.HAVE_IMU:
        from parts.imu import IMU

        imu = IMU(sensor=cfg.IMU_SENSOR, addr=cfg.IMU_ADDRESS,
                  dlp_setting=cfg.IMU_DLP_CONFIG)
        V.add(imu, outputs=['imu/acl_x', 'imu/acl_y', 'imu/acl_z',
                            'imu/gyr_x', 'imu/gyr_y', 'imu/gyr_z'], threaded=True)
    return imu

def add_drivetrain(V):
    from parts.actuator import VESC
    print("Creating VESC at port {}".format("/dev/ttyACM0"))
    vesc = VESC("/dev/ttyACM0")
    V.add(vesc, inputs=['steering', 'throttle'])

def parse_args() -> argparse.Namespace:
    """Configure and parse command-line options."""
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="Run DonkeyCar in various modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--js",
        action="store_true",
        help="utiliser une manette (joystick)",
    )
    parser.add_argument(
        "--type",
        choices=["tflite_linear", "tflite_imu"],
        default="tflite_linear",
        help="type de modèle à piloter",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config()
    drive(cfg, use_joystick=args.js, model_type=args.type)
