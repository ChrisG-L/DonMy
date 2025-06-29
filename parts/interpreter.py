import os
from abc import ABC, abstractmethod
import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorflow.python.framework.convert_to_constants import \
    convert_variables_to_constants_v2 as convert_var_to_const
from tensorflow.python.saved_model import tag_constants, signature_constants

class Interpreter(ABC):

    @abstractmethod
    def load(self, model_path):
        pass

    def load_weights(self, model_path, by_name = True):
        raise NotImplementedError('Requires implementation')

    def set_model(self, pilot):
        pass

    def set_optimizer(self, optimizer):
        pass

    def compile(self, **kwargs):
        raise NotImplementedError('Requires implementation')

    @abstractmethod
    def get_input_shapes(self):
        pass

    @abstractmethod
    def predict(self, img_arr, other_arr):
        pass

    def predict_from_dict(self, input_dict):
        pass

    def summary(self):
        pass

    def __str__(self):
        return type(self).__name__


class KerasInterpreter(Interpreter):

    def __init__(self):
        super().__init__()
        self.model = None

    def set_model(self, pilot):
        self.model = pilot.create_model()

    def set_optimizer(self, optimizer):
        self.model.optimizer = optimizer

    def get_input_shapes(self):
        assert self.model, 'Model not set'
        return [inp.shape for inp in self.model.inputs]

    def compile(self, **kwargs):
        assert self.model, 'Model not set'
        self.model.compile(**kwargs)

    def invoke(self, inputs):
        outputs = self.model(inputs, training=False)
        # for functional models the output here is a list
        if type(outputs) is list:
            # as we invoke the interpreter with a batch size of one we remove
            # the additional dimension here again
            output = [output.numpy().squeeze(axis=0) for output in outputs]
            return output
        # for sequential models the output shape is (1, n) with n = output dim
        else:
            return outputs.numpy().squeeze(axis=0)

    def predict(self, img_arr, other_arr) \
            :
        img_arr = np.expand_dims(img_arr, axis=0)
        inputs = img_arr
        if other_arr is not None:
            other_arr = np.expand_dims(other_arr, axis=0)
            inputs = [img_arr, other_arr]
        return self.invoke(inputs)

    def predict_from_dict(self, input_dict):
        for k, v in input_dict.items():
            input_dict[k] = np.expand_dims(v, axis=0)
        return self.invoke(input_dict)

    def load(self, model_path):
        print(f'Loading model {model_path}')
        self.model = keras.models.load_model(model_path, compile=False)

    def load_weights(self, model_path, by_name = True):
        assert self.model, 'Model not set'
        self.model.load_weights(model_path, by_name=by_name)

    def summary(self):
        return self.model.summary()


class TfLite(Interpreter):
    """
    This class wraps around the TensorFlow Lite interpreter.
    """

    def __init__(self):
        super().__init__()
        self.interpreter = None
        self.input_shapes = None
        self.input_details = None
        self.output_details = None

    def load(self, model_path):
        assert os.path.splitext(model_path)[1] == '.tflite', \
            'TFlitePilot should load only .tflite files'
        print(f'Loading model {model_path}')
        # Load TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Get Input shape
        self.input_shapes = []
        print('Load model with tflite input tensor details:')
        for detail in self.input_details:
            print(detail)
            self.input_shapes.append(detail['shape'])

    def compile(self, **kwargs):
        pass

    def invoke(self):
        self.interpreter.invoke()
        outputs = []
        for tensor in self.output_details:
            output_data = self.interpreter.get_tensor(tensor['index'])
            # as we invoke the interpreter with a batch size of one we remove
            # the additional dimension here again
            outputs.append(output_data[0])
        # don't return list if output is 1d
        return outputs if len(outputs) > 1 else outputs[0]

    def predict(self, img_arr, other_arr) \
            :
        assert self.input_shapes and self.input_details, \
            "Tflite model not loaded"
        input_arrays = (img_arr, other_arr)
        for arr, shape, detail \
                in zip(input_arrays, self.input_shapes, self.input_details):
            in_data = arr.reshape(shape).astype(np.float32)
            self.interpreter.set_tensor(detail['index'], in_data)
        return self.invoke()

    def predict_from_dict(self, input_dict):
        for detail in self.input_details:
            k = detail['name']
            inp_k = input_dict[k]
            inp_k_res = inp_k.reshape(detail['shape']).astype(np.float32)
            self.interpreter.set_tensor(detail['index'], inp_k_res)
        return self.invoke()

    def get_input_shapes(self):
        print("\n\n\n\n\nget_input_shapes Inside true\n\n\n\n\n")
        assert self.input_shapes is not None, "Need to load model first"
        return self.input_shapes


class TensorRT(Interpreter):
    """
    Uses TensorRT to do the inference.
    """
    def __init__(self):
        self.frozen_func = None
        self.input_shapes = None

    def get_input_shapes(self):
        return self.input_shapes

    def compile(self, **kwargs):
        pass

    def load(self, model_path):
        saved_model_loaded = tf.saved_model.load(model_path,
                                                 tags=[tag_constants.SERVING])
        graph_func = saved_model_loaded.signatures[
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.frozen_func = convert_var_to_const(graph_func)
        self.input_shapes = [inp.shape for inp in graph_func.inputs]

    def predict(self, img_arr, other_arr) \
            :
        # first reshape as usual
        img_arr = np.expand_dims(img_arr, axis=0).astype(np.float32)
        img_tensor = self.convert(img_arr)
        if other_arr is not None:
            other_arr = np.expand_dims(other_arr, axis=0).astype(np.float32)
            other_tensor = self.convert(other_arr)
            output_tensors = self.frozen_func(img_tensor, other_tensor)
        else:
            output_tensors = self.frozen_func(img_tensor)

        # because we send a batch of size one, pick first element
        outputs = [out.numpy().squeeze(axis=0) for out in output_tensors]
        # don't return list if output is 1d
        return outputs if len(outputs) > 1 else outputs[0]

    def predict_from_dict(self, input_dict):
        args = []
        for inp in self.frozen_func.inputs:
            name = inp.name.split(':')[0]
            val = input_dict[name]
            val_res = np.expand_dims(val, axis=0).astype(np.float32)
            val_conv = self.convert(val_res)
            args.append(val_conv)
        output_tensors = self.frozen_func(*args)
        # because we send a batch of size one, pick first element
        outputs = [out.numpy().squeeze(axis=0) for out in output_tensors]
        # don't return list if output is 1d
        return outputs if len(outputs) > 1 else outputs[0]

    @staticmethod
    def convert(arr):
        """ Helper function. """
        value = tf.compat.v1.get_variable("features", dtype=tf.float32,
                                          initializer=tf.constant(arr))
        return tf.convert_to_tensor(value=value)
