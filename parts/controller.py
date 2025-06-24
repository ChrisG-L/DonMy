import os
import array
import time
import struct


class Joystick(object):
    # États d'arrêt d'urgence
    ES_IDLE = -1
    ES_START = 0
    ES_THROTTLE_NEG_ONE = 1
    ES_THROTTLE_POS_ONE = 2
    ES_THROTTLE_NEG_TWO = 3

    def __init__(self,
                 dev_fn='/dev/input/js0',
                 poll_delay=0.0,
                 throttle_scale=0.05,
                 steering_scale=1.0,
                 throttle_dir=-1.0,
                 auto_record_on_throttle=True):

        # Paramètres du joystick
        self.dev_fn = dev_fn
        self.jsdev = None
        self.js_name = ""
        self.num_axes = 0
        self.num_buttons = 0

        # États des axes et boutons
        self.axis_states = {}
        self.button_states = {}
        self.axis_map = []
        self.button_map = []

        # Configuration Logitech
        self.axis_names = {
            0x00: 'left_stick_horz',
            0x01: 'left_stick_vert',
            0x03: 'right_stick_horz',
            0x04: 'right_stick_vert',
            0x02: 'L2_pressure',
            0x05: 'R2_pressure',
            0x10: 'dpad_leftright',
            0x11: 'dpad_up_down',
        }

        self.button_names = {
            0x13a: 'back',
            0x13b: 'start',
            0x13c: 'Logitech',
            0x130: 'A',
            0x131: 'B',
            0x133: 'X',
            0x134: 'Y',
            0x136: 'L1',
            0x137: 'R1',
            0x13d: 'left_stick_press',
            0x13e: 'right_stick_press',
        }

        # Paramètres du contrôleur
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
        self.tub = None
        self.num_records_to_erase = 100
        self.estop_state = self.ES_IDLE
        self.dead_zone = 0.01

        # Maps des triggers
        self.button_down_trigger_map = {
            'Y': self.erase_last_N_records,
            'A': self.emergency_stop,
        }

        self.axis_trigger_map = {
            'left_stick_horz': self.set_steering,
            'R2_pressure': self.set_throttle,
            'L2_pressure': self.set_throttle_back,
        }

    def init_joystick(self):
        """Initialise le joystick"""
        try:
            from fcntl import ioctl
        except ModuleNotFoundError:
            self.num_axes = 0
            self.num_buttons = 0
            print("Pas de support pour le module fcntl. Joystick désactivé.")
            return False

        if not os.path.exists(self.dev_fn):
            print(f"{self.dev_fn} manquant")
            return False

        try:
            # Ouvrir le périphérique joystick
            print(f'Ouverture de {self.dev_fn}...')
            self.jsdev = open(self.dev_fn, 'rb')

            # Obtenir le nom du périphérique
            buf = array.array('B', [0] * 64)
            ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)
            self.js_name = buf.tobytes().decode('utf-8')
            print('Nom du périphérique: %s' % self.js_name)

            # Obtenir le nombre d'axes et de boutons
            buf = array.array('B', [0])
            ioctl(self.jsdev, 0x80016a11, buf)
            self.num_axes = buf[0]

            buf = array.array('B', [0])
            ioctl(self.jsdev, 0x80016a12, buf)
            self.num_buttons = buf[0]

            # Obtenir la carte des axes
            buf = array.array('B', [0] * 0x40)
            ioctl(self.jsdev, 0x80406a32, buf)

            for axis in buf[:self.num_axes]:
                axis_name = self.axis_names.get(axis, 'unknown(0x%02x)' % axis)
                self.axis_map.append(axis_name)
                self.axis_states[axis_name] = 0.0

            # Obtenir la carte des boutons
            buf = array.array('H', [0] * 200)
            ioctl(self.jsdev, 0x80406a34, buf)

            for btn in buf[:self.num_buttons]:
                btn_name = self.button_names.get(btn, 'unknown(0x%03x)' % btn)
                self.button_map.append(btn_name)
                self.button_states[btn_name] = 0

            return True

        except FileNotFoundError:
            print(f"{self.dev_fn} non trouvé.")
            return False
        except Exception as e:
            print(f"Erreur lors de l'initialisation du joystick: {e}")
            return False

    def poll_joystick(self):
        """Interroge le joystick pour les événements d'entrée"""
        button = None
        button_state = None
        axis = None
        axis_val = None

        if self.jsdev is None:
            return button, button_state, axis, axis_val

        try:
            evbuf = self.jsdev.read(8)
            if evbuf:
                tval, value, typev, number = struct.unpack('IhBB', evbuf)

                if typev & 0x80:
                    # Ignorer l'événement d'initialisation
                    return button, button_state, axis, axis_val

                if typev & 0x01:
                    button = self.button_map[number]
                    if button:
                        self.button_states[button] = value
                        button_state = value
                        print("bouton: %s état: %d" % (button, value))

                if typev & 0x02:
                    axis = self.axis_map[number]
                    if axis:
                        fvalue = value / 32767.0
                        self.axis_states[axis] = fvalue
                        axis_val = fvalue
                        print("axe: %s val: %f" % (axis, fvalue))

        except Exception as e:
            print(f"Erreur lors de la lecture du joystick: {e}")

        return button, button_state, axis, axis_val

    def set_tub(self, tub):
        """Définit le conteneur de données"""
        self.tub = tub

    def erase_last_N_records(self):
        """Efface les N derniers enregistrements"""
        if self.tub is not None:
            try:
                self.tub.delete_last_n_records(self.num_records_to_erase)
                print('Supprimé les %d derniers enregistrements.' % self.num_records_to_erase)
            except:
                print('Échec de l\'effacement')

    def on_throttle_changes(self):
        """Active l'enregistrement quand l'accélération est non nulle en mode utilisateur"""
        if self.auto_record_on_throttle:
            recording = (abs(self.throttle) > self.dead_zone and self.mode == 'user')
            if recording != self.recording:
                self.recording = recording
                self.recording_latch = self.recording
                print(f"Enregistrement défini à: {self.recording}")

    def emergency_stop(self):
        """Initie un arrêt d'urgence"""
        print('Arrêt d\'urgence!')
        self.mode = "user"
        self.recording = False
        self.constant_throttle = False
        self.estop_state = self.ES_START
        self.throttle = 0.0

    def set_steering(self, axis_val):
        """Définit l'angle de direction"""
        self.angle = self.steering_scale * axis_val

    def set_throttle(self, axis_val):
        """Définit l'accélération avant"""
        self.last_throttle_axis_val = axis_val
        temp = (self.throttle_dir * axis_val)
        if temp >= 0:
            temp = 0
        self.throttle = max(-0.30, temp)
        self.on_throttle_changes()

    def set_throttle_back(self, axis_val):
        """Définit l'accélération arrière"""
        self.last_throttle_axis_val = axis_val
        temp = (self.throttle_dir * axis_val)
        if temp <= 0:
            temp = 0
        self.throttle = min(0.30, temp)
        self.on_throttle_changes()
        print(f'enregistrement: {self.recording}')

    def update(self):
        """Boucle principale de mise à jour"""
        # Attendre que le joystick soit en ligne
        while self.running and not self.init_joystick():
            time.sleep(3)

        while self.running:
            button, button_state, axis, axis_val = self.poll_joystick()

            if axis is not None and axis in self.axis_trigger_map:
                self.axis_trigger_map[axis](axis_val)

            if button and button_state >= 1 and button in self.button_down_trigger_map:
                self.button_down_trigger_map[button]()

            time.sleep(self.poll_delay)

    def run_threaded(self, img_arr=None, mode=None, recording=None):
        """Méthode principale d'exécution"""
        self.img_arr = img_arr

        # Appliquer les valeurs par défaut si elles ne sont pas None
        if mode is not None:
            self.mode = mode
        if self.mode_latch is not None:
            self.mode = self.mode_latch
            self.mode_latch = None
        if recording is not None and recording != self.recording:
            self.recording = recording
        if self.recording_latch is not None:
            self.recording = self.recording_latch
            self.recording_latch = None

        # Traiter la machine à états d'arrêt d'urgence
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
        """Alias pour run_threaded"""
        return self.run_threaded(img_arr, mode, recording)

    def shutdown(self):
        """Arrêt propre du contrôleur"""
        self.running = False
        time.sleep(0.5)
        if self.jsdev:
            self.jsdev.close()

def get_js_controller(cfg=None):
    """
    Fonction utilitaire pour créer et configurer un contrôleur de joystick
    """
    ctr = Joystick()
    return ctr
