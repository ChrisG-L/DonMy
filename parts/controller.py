import os
import array
import time
import struct

class Joystick(object):
    def __init__(self, dev_fn='/dev/input/js0'):
        self.axis_states = {}
        self.button_states = {}
        self.axis_names = {}
        self.button_names = {}
        self.axis_map = []
        self.button_map = []
        self.jsdev = None
        self.dev_fn = dev_fn


    def init(self):
        try:
            from fcntl import ioctl
        except ModuleNotFoundError:
            self.num_axes = 0
            self.num_buttons = 0
            print("no support for fnctl module. joystick not enabled.")
            return False

        if not os.path.exists(self.dev_fn):
            print(f"{self.dev_fn} is missing")
            return False

        # Open the joystick device.
        print(f'Opening %s... {self.dev_fn}')
        self.jsdev = open(self.dev_fn, 'rb')

        # Get the device name.
        buf = array.array('B', [0] * 64)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf) # JSIOCGNAME(len)
        self.js_name = buf.tobytes().decode('utf-8')
        print('Device name: %s' % self.js_name)

        # Get number of axes and buttons.
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf) # JSIOCGAXES
        self.num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf) # JSIOCGBUTTONS
        self.num_buttons = buf[0]

        # Get the axis map.
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf) # JSIOCGAXMAP

        for axis in buf[:self.num_axes]:
            axis_name = self.axis_names.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf) # JSIOCGBTNMAP

        for btn in buf[:self.num_buttons]:
            btn_name = self.button_names.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0
            #print('btn', '0x%03x' % btn, 'name', btn_name)

        return True


    def poll(self):
        button = None
        button_state = None
        axis = None
        axis_val = None

        if self.jsdev is None:
            return button, button_state, axis, axis_val

        # Main event loop
        evbuf = self.jsdev.read(8)

        if evbuf:
            tval, value, typev, number = struct.unpack('IhBB', evbuf)

            if typev & 0x80:
                #ignore initialization event
                return button, button_state, axis, axis_val

            if typev & 0x01:
                button = self.button_map[number]
                #print(tval, value, typev, number, button, 'pressed')
                if button:
                    self.button_states[button] = value
                    button_state = value
                    print("button: %s state: %d" % (button, value))

            if typev & 0x02:
                axis = self.axis_map[number]
                if axis:
                    fvalue = value / 32767.0
                    self.axis_states[axis] = fvalue
                    axis_val = fvalue
                    print("axis: %s val: %f" % (axis, fvalue))

        return button, button_state, axis, axis_val

class LogitechJoystick(Joystick):
    def __init__(self, *args, **kwargs):
        print("\n\n\n\n\n\n\n\n\n\nLOGITECHHHHH\n\n\n\n\n\n")
        super(LogitechJoystick, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00: 'left_stick_horz',
            0x01: 'left_stick_vert',
            0x03: 'right_stick_horz',
            0x04: 'right_stick_vert',

            0x02: 'L2_pressure',
            0x05: 'R2_pressure',

            0x10: 'dpad_leftright', # 1 is right, -1 is left
            0x11: 'dpad_up_down', # 1 is down, -1 is up
        }

        self.button_names = {
            0x13a: 'back',  # 8 314
            0x13b: 'start',  # 9 315
            0x13c: 'Logitech',  # a  316

            0x130: 'A',
            0x131: 'B',
            0x133: 'X',
            0x134: 'Y',

            0x136: 'L1',
            0x137: 'R1',

            0x13d: 'left_stick_press',
            0x13e: 'right_stick_press',
        }

class JoystickController(object):
    ES_IDLE = -1
    ES_START = 0
    ES_THROTTLE_NEG_ONE = 1
    ES_THROTTLE_POS_ONE = 2
    ES_THROTTLE_NEG_TWO = 3


    def __init__(self, poll_delay=0.0,
                 throttle_scale=0.05,
                 steering_scale=1.0,
                 throttle_dir=-1.0,
                 dev_fn='/dev/input/js0',
                 auto_record_on_throttle=True):

        self.img_arr = None
        self.angle = 0.0
        self.throttle = 0.0
        self.mode = 'user'
        self.mode_latch = None
        self.poll_delay = poll_delay
        self.running = True
        self.last_throttle_axis_val = 0
        self.throttle_scale = throttle_scale
        self.steering_scale = steering_scale
        self.throttle_dir = throttle_dir
        self.recording = False
        self.recording_latch = None
        self.constant_throttle = False
        self.auto_record_on_throttle = auto_record_on_throttle
        self.dev_fn = dev_fn
        self.js = None
        self.tub = None
        self.num_records_to_erase = 100
        self.estop_state = self.ES_IDLE
        self.dead_zone = 0.0

        self.button_down_trigger_map = {}
        self.axis_trigger_map = {}
        self.init_trigger_maps()


    def init_js(self):
        raise(Exception("Subclass needs to define init_js"))


    def set_deadzone(self, val):
        '''
        sets the minimim throttle for recording
        '''
        self.dead_zone = val


    def set_tub(self, tub):
        self.tub = tub


    def erase_last_N_records(self):
        if self.tub is not None:
            try:
                self.tub.delete_last_n_records(self.num_records_to_erase)
                print('deleted last %d records.' % self.num_records_to_erase)
            except:
                print('failed to erase')


    def on_throttle_changes(self):
        '''
        turn on recording when non zero throttle in the user mode.
        '''
        if self.auto_record_on_throttle:
            recording = (abs(self.throttle) > self.dead_zone and self.mode == 'user')
            if recording != self.recording:
                self.recording = recording
                self.recording_latch = self.recording
                print(f"JoystickController::on_throttle_changes() setting recording = {self.recording}")


    def emergency_stop(self):
        '''
        initiate a series of steps to try to stop the vehicle as quickly as possible
        '''
        print('E-Stop!!!')
        self.mode = "user"
        self.recording = False
        self.constant_throttle = False
        self.estop_state = self.ES_START
        self.throttle = 0.0


    def update(self):
        '''
        poll a joystick for input events
        '''

        #wait for joystick to be online
        while self.running and self.js is None and not self.init_js():
            time.sleep(3)

        while self.running:
            button, button_state, axis, axis_val = self.js.poll()

            if axis is not None and axis in self.axis_trigger_map:
                '''
                then invoke the function attached to that axis
                '''
                self.axis_trigger_map[axis](axis_val)

            if button and button_state >= 1 and button in self.button_down_trigger_map:
                '''
                then invoke the function attached to that button
                '''
                self.button_down_trigger_map[button]()

            time.sleep(self.poll_delay)

    def set_steering(self, axis_val):
        self.angle = self.steering_scale * axis_val
        #print("angle", self.angle)


    def set_throttle(self, axis_val):
        #this value is often reversed, with positive value when pulling down
        self.last_throttle_axis_val = axis_val
        temp = (self.throttle_dir * axis_val)
        #print("\t\tFFthrottle", temp)
        if temp >= 0:
            temp = 0
        self.throttle = max(-0.30, temp)
        self.on_throttle_changes()
    
    def set_throttle_back(self, axis_val):
        #this value is often reversed, with positive value when pulling down
        self.last_throttle_axis_val = axis_val
        temp = (self.throttle_dir * axis_val)
        #print("Bthrottle", temp)
        if temp <= 0:
            temp = 0
        self.throttle = min(0.30, temp)
        self.on_throttle_changes()


        print(f'recording: {self.recording}')


    def toggle_constant_throttle(self):
        '''
        toggle constant throttle
        '''
        if self.constant_throttle:
            self.constant_throttle = False
            self.throttle = 0
            self.on_throttle_changes()
        else:
            self.constant_throttle = True
            self.throttle = self.throttle_scale
            self.on_throttle_changes()
        print(f'constant_throttle: {self.constant_throttle}')


    def run_threaded(self, img_arr=None, mode=None, recording=None):
        """
        :param img_arr: current camera image or None
        :param mode: default user/mode
        :param recording: default recording mode
        """
        self.img_arr = img_arr

        #
        # enforce defaults if they are not none.
        #
        if mode is not None:
            self.mode = mode
        if self.mode_latch is not None:
            self.mode = self.mode_latch
            self.mode_latch = None
        if recording is not None and recording != self.recording:
            print(f"JoystickController::run_threaded() setting recording from default = {recording}")
            self.recording = recording
        if self.recording_latch is not None:
            print(f"JoystickController::run_threaded() setting recording from latch = {self.recording_latch}")
            self.recording = self.recording_latch
            self.recording_latch = None

        '''
        process E-Stop state machine
        '''
        if self.estop_state > self.ES_IDLE:
            if self.estop_state == self.ES_START:
                self.estop_state = self.ES_THROTTLE_NEG_ONE
                return 0.0, -1.0 * self.throttle_scale, self.mode, False
            elif self.estop_state == self.ES_THROTTLE_NEG_ONE:
                self.estop_state = self.ES_THROTTLE_POS_ONE
                return 0.0, 0.01, self.mode, False
            elif self.estop_state == self.ES_THROTTLE_POS_ONE:
                self.estop_state = self.ES_THROTTLE_NEG_TWO
                self.throttle = -1.0 * self.throttle_scale
                return 0.0, self.throttle, self.mode, False
            elif self.estop_state == self.ES_THROTTLE_NEG_TWO:
                self.throttle += 0.05
                if self.throttle >= 0.0:
                    self.throttle = 0.0
                    self.estop_state = self.ES_IDLE
                return 0.0, self.throttle, self.mode, False


        return self.angle, self.throttle, self.mode, self.recording


    def run(self, img_arr=None, mode=None, recording=None):
        return self.run_threaded(img_arr, mode, recording)


    def shutdown(self):
        #set flag to exit polling thread, then wait a sec for it to leave
        self.running = False
        time.sleep(0.5)

class LogitechJoystickController(JoystickController):
    def __init__(self, *args, **kwargs):
        super(LogitechJoystickController, self).__init__(*args, **kwargs)


    def init_js(self):
        '''
        attempt to init joystick
        '''
        try:
            self.js = LogitechJoystick(self.dev_fn)
            self.js.init()
        except FileNotFoundError:
            print(f"{self.dev_fn} not found.")
            self.js = None
        return self.js is not None


    def init_trigger_maps(self):
        '''
        init set of mapping from buttons to function calls
        '''

        self.button_down_trigger_map = {
            'Y': self.erase_last_N_records,
            'A': self.emergency_stop,
            'back': self.toggle_constant_throttle,
        }

        self.axis_trigger_map = {
            'left_stick_horz': self.set_steering,
            'R2_pressure': self.set_throttle,
            'L2_pressure': self.set_throttle_back,
        }

def get_js_controller(cfg):
    ctr = LogitechJoystickController()

    ctr.set_deadzone(0.01)
    return ctr
