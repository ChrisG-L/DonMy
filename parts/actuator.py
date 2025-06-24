"""
actuators.py
Classes to control the motors and servos. These classes
are wrapped in a mixer class before being used in the drive loop.
"""

import time

class VESC:
    def __init__(self, serial_port, percent=.2, has_sensor=True, start_heartbeat=True, baudrate=115200, timeout=0.05, angle_scale = 0.5, angle_offset = 0.5 ):

        import pyvesc

        self.angle_scale = angle_scale
        self.angle_offset = angle_offset
        self.percent = percent

        try:
            self.v = pyvesc.VESC(serial_port, has_sensor, start_heartbeat, baudrate, timeout)
        except Exception as err:
            print("\n\n\n\n", err)
            print("\n\nto fix permission denied errors, try running the following command:")
            print("sudo chmod a+rw {}".format(serial_port), "\n\n\n\n")
            time.sleep(1)
            raise

    def run(self, angle, throttle):
        self.v.set_servo((angle * self.angle_scale) + self.angle_offset)
        # print(f"\tThro : {throttle} | Perc : {self.percent} ==> {throttle*self.percent}")
        self.v.set_duty_cycle(throttle*self.percent)
