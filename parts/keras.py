import datetime
from abc import ABC, abstractmethod

import numpy as np

from tensorflow.python.data.ops.dataset_ops import DatasetV1, DatasetV2

from main_parts.utils import normalize_image
from main_parts.types import TubRecord
from parts.interpreter import Interpreter, KerasInterpreter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D, BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import deque

class KerasPilot(ABC):
    def __init__(self,
                 interpreter = KerasInterpreter(),
                 input_shape = (120, 160, 3)):
        self.input_shape = input_shape
        self.optimizer = "adam"
        self.interpreter = interpreter
        self.interpreter.set_model(self)
        print(f'Created {self} with interpreter: {interpreter}')

    def load(self, model_path):
        print(f'Loading model {model_path}')
        self.interpreter.load(model_path)

    def load_weights(self, model_path, by_name = True):
        self.interpreter.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay):
        if optimizer_type == "adam":
            optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception(f"Unknown optimizer type: {optimizer_type}")
        self.interpreter.set_optimizer(optimizer)

    def get_input_shapes(self):
        return self.interpreter.get_input_shapes()

    def seq_size(self):
        return 0

    def run(self, img_arr, other_arr = None):
        norm_arr = normalize_image(img_arr)
        np_other_array = np.array(other_arr) if other_arr else None
        return self.inference(norm_arr, np_other_array)

    def inference(self, img_arr, other_arr):
        out = self.interpreter.predict(img_arr, other_arr)
        return self.interpreter_to_output(out)

    def inference_from_dict(self, input_dict):
        output = self.interpreter.predict_from_dict(input_dict)
        return self.interpreter_to_output(output)

    @abstractmethod
    def interpreter_to_output(self, interpreter_out):
        pass

    def train(self,
              model_path,
              train_data,
              train_steps,
              batch_size,
              validation_data,
              validation_steps,
              epochs,
              verbose = 1,
              min_delta = .0005,
              patience = 5):
        model = self.interpreter.model
        self.compile()

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=patience,
                          min_delta=min_delta),
            ModelCheckpoint(monitor='val_loss',
                            filepath=model_path,
                            save_best_only=True,
                            verbose=verbose)]

        tic = datetime.datetime.now()
        print('////////// Starting training //////////')
        history = model.fit(
            x=train_data,
            steps_per_epoch=train_steps,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=verbose,
            workers=1,
            use_multiprocessing=False)
        toc = datetime.datetime.now()
        print(f'////////// Finished training in: {toc - tic} //////////')
        return history.history

    def x_transform(self, record, img_processor):
        img_arr = record.image(processor=img_processor)
        return {'img_in': img_arr}

    def y_transform(self, record = None):
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def output_types(self):
        shapes = self.output_shapes()
        types = tuple({k: tf.float64 for k in d} for d in shapes)
        return types

    def output_shapes(self):
        return {}

    def __str__(self):
        return type(self).__name__

class KerasLinear(KerasPilot):
    def __init__(self,
                 interpreter = KerasInterpreter(),
                 input_shape = (120, 160, 3),
                 num_outputs = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(self.num_outputs, self.input_shape)

    def compile(self):
        self.interpreter.compile(optimizer=self.optimizer, loss='mse')

    def interpreter_to_output(self, interpreter_out):
        angle = interpreter_out[0]
        throttle = interpreter_out[1]
        return angle[0], throttle[0]

    def y_transform(self, record):
        angle = record.underlying['user/angle']
        throttle = record.underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        img_shape = self.get_input_shape('img_in')[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes

class KerasMemory(KerasLinear):
    def __init__(self,
                 interpreter = KerasInterpreter(),
                 input_shape = (120, 160, 3),
                 mem_length = 3,
                 mem_depth = 0,
                 mem_start_speed = 0.0):
        self.mem_length = mem_length
        self.mem_start_speed = mem_start_speed
        self.mem_seq = deque([[0, mem_start_speed]] * mem_length)
        self.mem_depth = mem_depth
        super().__init__(interpreter, input_shape)

    def seq_size(self):
        return self.mem_length + 1

    def create_model(self):
        return default_memory(self.input_shape,
                              self.mem_length, self.mem_depth)

    def load(self, model_path):
        super().load(model_path)
        self.mem_length = self.interpreter.get_input_shapes()[1][1] // 2
        self.mem_seq = deque([[0, self.mem_start_speed]] * self.mem_length)
        print(f'Loaded memory model with mem length {self.mem_length}')

    def run(self, img_arr, other_arr):
        np_mem_arr = np.array(self.mem_seq).reshape((2 * self.mem_length,))
        img_arr_norm = normalize_image(img_arr)
        angle, throttle = super().inference(img_arr_norm, np_mem_arr)
        self.mem_seq.popleft()
        self.mem_seq.append([angle, throttle])
        return angle, throttle

    def x_transform(
            self,
            record,
            img_processor):
        assert isinstance(record, list), 'List[TubRecord] expected'
        assert len(record) == self.mem_length + 1, \
            f"Record list of length {self.mem_length} required but " \
            f"{len(record)} was passed"
        img_arr = record[-1].image(processor=img_processor)
        mem = [[r.underlying['user/angle'], r.underlying['user/throttle']]
               for r in record[:-1]]
        np_mem = np.array(mem).reshape((2 * self.mem_length,))
        return {'img_in': img_arr, 'mem_in': np_mem}

    def y_transform(self, records):
        assert isinstance(records, list), 'List[TubRecord] expected'
        angle = records[-1].underlying['user/angle']
        throttle = records[-1].underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'mem_in': tf.TensorShape(2 * self.mem_length)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes

    def __str__(self):
        return super().__str__() \
            + f'-L:{self.mem_length}-D:{self.mem_depth}'

def default_memory(input_shape=(120, 160, 3), mem_length=3, mem_depth=0):
    drop = 0.2
    drop2 = 0.1
    print(f'Creating memory model with length {mem_length}, depth '
                f'{mem_depth}')
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    mem_in = Input(shape=(2 * mem_length,), name='mem_in')
    y = mem_in
    for i in range(mem_depth):
        y = Dense(4 * mem_length, activation='relu', name=f'mem_{i}')(y)
        y = Dropout(drop2)(y)
    for i in range(1, mem_length):
        y = Dense(2 * (mem_length - i), activation='relu', name=f'mem_c_{i}')(y)
        y = Dropout(drop2)(y)
    x = concatenate([x, y])
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)
    activation = ['tanh', 'sigmoid']
    outputs = [Dense(1, activation=activation[i], name='n_outputs' + str(i))(x)
               for i in range(2)]
    model = Model(inputs=[img_in, mem_in], outputs=outputs, name='memory')
    return model

def conv2d(filters, kernel, strides, layer_num, activation='relu'):
    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides, strides),
                         activation=activation,
                         name='conv2d_' + str(layer_num))

def core_cnn_layers(img_in, drop, l4_stride=1):
    x = img_in
    x = conv2d(24, 5, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(32, 5, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 5, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, l4_stride, 4)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 1, 5)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    return x

def default_n_linear(num_outputs, input_shape=(120, 160, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    x = Dense(100, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(50, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(
            Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name='linear')
    return model
