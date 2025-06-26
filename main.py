import argparse

import main_parts.config as cfg
from main_parts.vehicle import Vehicle
from main_parts.utils import get_model
from parts.classUtils import ToggleRecording, DriveMode, UserPilotCondition

from parts.oakd_camera import OakDCamera
from parts.tub_v2 import TubWriter

def drive(model_path="./models/mypilot.tflite", use_joystick=False):
    V = Vehicle()

    cam = OakDCamera(width=cfg.IMAGE_W,
            height=cfg.IMAGE_H,
            fps=cfg.CAMERA_FRAMERATE)
    V.add(cam, inputs=[], outputs=['cam/image_array'], threaded=True)

    ctr = add_user_controller(V, use_joystick)

    V.add(UserPilotCondition(show_pilot_image=False),
        inputs=['user/mode', "cam/image_array", "cam/image_array_trans"],
        outputs=['run_user', "run_pilot", "ui/image_array"])

    def load_model(kl, model_path):
        print('\nloading model', model_path)
        kl.load(model_path)
        print('model loaded\n')

    if not use_joystick:
        kl = get_model(False)

        load_model(kl, model_path)

        inputs = ['cam/image_array']

        outputs = ['pilot/angle', 'pilot/throttle']

        V.add(kl, inputs=inputs, outputs=outputs, run_condition='run_pilot')

    V.add(DriveMode(),
          inputs=['user/mode', 'user/angle', 'user/throttle',
                  'pilot/angle', 'pilot/throttle'],
          outputs=['angle', 'throttle'])

    recording_control = ToggleRecording()
    V.add(recording_control, inputs=['user/mode', "recording"], outputs=["recording"])

    add_drivetrain(V)

    inputs=['cam/image_array','user/angle', 'user/throttle', 'user/mode']
    types=['image_array','float', 'float','str']

    tub_path = cfg.DATA_PATH
    tub_writer = TubWriter(tub_path, inputs=inputs, types=types)
    V.add(tub_writer, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    if ctr is not None:
        print("You can now move your controller to drive your car.")
        ctr.set_tub(tub_writer.tub)

    V.start(rate_hz=cfg.DRIVE_LOOP_HZ)

def add_user_controller(V, use_joystick, input_image='ui/image_array'):
    ctr = None
    if use_joystick:
        from parts.controller import Joystick
        ctr = Joystick()
        V.add(
            ctr,
            inputs=[input_image, 'user/mode', 'recording'],
            outputs=['user/angle', 'user/throttle',
                        'user/mode', 'recording'],
            threaded=True)
    return ctr

def add_drivetrain(V):
    from parts.actuator import VESC
    vesc = None
    try:
        vesc = VESC("/dev/ttyACM1")
    except:
        try:
            vesc = VESC("/dev/ttyACM0")
        except Exception as e:
            print(f"Error /dev/tty: {e}")
    V.add(vesc, inputs=['angle', 'throttle'])

def parse_args():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Run in various modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--js",
        action="store_true",
        help="utiliser une manette (joystick)",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    drive(use_joystick=args.js)
