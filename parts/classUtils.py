# -*- coding: utf-8 -*-

import time

class Pipe:
    def run(self, *args):
        # seems to be a python bug that takes a single argument
        # return makes it into two element tuple with empty last element.
        return args if len(args) > 1 else args[0]

class ToggleRecording:
    def __init__(self, auto_record_on_throttle=True, record_in_autopilot=False):
        self.auto_record_on_throttle = auto_record_on_throttle
        self.record_in_autopilot = record_in_autopilot
        self.recording_latch: bool = None
        self.toggle_latch: bool = False
        self.last_recording = None

    def run(self, mode: str, recording: bool):
        recording_in = recording
        if recording_in != self.last_recording:
            print(f"Recording Change = {recording_in}")

        if self.toggle_latch:
            if self.auto_record_on_throttle:
                print(
                    'auto record on throttle is enabled; ignoring toggle of manual mode.')
            else:
                recording = not self.last_recording
            self.toggle_latch = False

        if self.recording_latch is not None:
            recording = self.recording_latch
            self.recording_latch = None

        if recording and mode != 'user' and not self.record_in_autopilot:
            print("Ignoring recording in auto-pilot mode")
            recording = False

        if self.last_recording != recording:
            print(f"Setting Recording = {recording}")

        self.last_recording = recording

        return recording

class DriveMode:
    def __init__(self, ai_throttle_mult=1.0):
        self.ai_throttle_mult = ai_throttle_mult

    def run(self, mode,
            user_steering, user_throttle,
            pilot_steering, pilot_throttle):
        if mode == 'user':
            return user_steering, user_throttle
        elif mode == 'local_angle':
            return pilot_steering if pilot_steering else 0.0, user_throttle
        return (pilot_steering if pilot_steering else 0.0,
               pilot_throttle * self.ai_throttle_mult if pilot_throttle else 0.0)

class TriggeredCallback:
    def __init__(self, args, func_cb):
        self.args = args
        self.func_cb = func_cb

    def run(self, trigger):
        if trigger:
            self.func_cb(self.args)

    def shutdown(self):
        return

class DelayedTrigger:
    def __init__(self, delay):
        self.ticks = 0
        self.delay = delay

    def run(self, trigger):
        if self.ticks > 0:
            self.ticks -= 1
            if self.ticks == 0:
                return True

        if trigger:
            self.ticks = self.delay

        return False

    def shutdown(self):
        return
