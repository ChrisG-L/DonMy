import time
from threading import Thread
from .memory import Memory
import traceback

class Vehicle:
    def __init__(self, mem=None):
        if not mem:
            mem = Memory()
        self.mem = mem
        self.parts = []
        self.on = True
        self.threads = []

    def add(self, part, inputs=[], outputs=[],
            threaded=False, run_condition=None):
        assert type(inputs) is list, "inputs is not a list: %r" % inputs
        assert type(outputs) is list, "outputs is not a list: %r" % outputs
        assert type(threaded) is bool, "threaded is not a boolean: %r" % threaded

        p = part
        entry = {}
        entry['part'] = p
        entry['inputs'] = inputs
        entry['outputs'] = outputs
        entry['run_condition'] = run_condition

        if threaded:
            t = Thread(target=part.update, args=())
            t.daemon = True
            entry['thread'] = t

        self.parts.append(entry)

    def start(self, rate_hz=20):
        try:
            self.on = True

            for entry in self.parts:
                if entry.get('thread'):
                    entry.get('thread').start()

            loop_start_time = time.time()
            loop_count = 0
            while self.on:
                start_time = time.time()
                loop_count += 1

                self.update_parts()

                sleep_time = 1.0 / rate_hz - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

            loop_total_time = time.time() - loop_start_time

            return loop_count, loop_total_time

        except KeyboardInterrupt:
            pass
        except Exception as e:
            traceback.print_exc()
        finally:
            self.stop()

    def update_parts(self):
        for entry in self.parts:

            run = True
            if entry.get('run_condition'):
                run_condition = entry.get('run_condition')
                run = self.mem.get([run_condition])[0]

            if run:
                p = entry['part']
                inputs = self.mem.get(entry['inputs'])
                if entry.get('thread'):
                    outputs = p.run_threaded(*inputs)
                else:
                    outputs = p.run(*inputs)

                if outputs is not None:
                    self.mem.put(entry['outputs'], outputs)

    def stop(self):
        for entry in self.parts:
            try:
                entry['part'].shutdown()
            except AttributeError:
                pass
            except Exception as e:
                print(f"Error: {e}")
