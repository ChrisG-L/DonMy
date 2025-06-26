class UserPilotCondition:
    def __init__(self, show_pilot_image=False):
        self.show_pilot_image = show_pilot_image

    def run(self, mode, user_image, pilot_image):
        if mode == 'user':
            return True, False, user_image
        else:
            return False, True, pilot_image if self.show_pilot_image else user_image

class ToggleRecording:
    def __init__(self, auto_record_on_throttle=True, record_in_autopilot=False):
        self.auto_record_on_throttle = auto_record_on_throttle
        self.record_in_autopilot = record_in_autopilot
        self.recording_latch = None
        self.toggle_latch = False
        self.last_recording = None

    def run(self, mode, recording):
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
            user_angle, user_throttle,
            pilot_angle, pilot_throttle):
        if mode == 'user':
            return user_angle, user_throttle
        elif mode == 'local_angle':
            return pilot_angle if pilot_angle else 0.0, user_throttle
        return (pilot_angle if pilot_angle else 0.0,
               pilot_throttle * self.ai_throttle_mult if pilot_throttle else 0.0)
