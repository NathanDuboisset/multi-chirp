import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

import glob
from tensorflow import keras as k
from keras import layers as l


INTERPRETER = None

def loadModel(model_path):

    global INTERPRETER
    global INPUT_LAYER_INDEX
    global OUTPUT_LAYER_INDEX

    # Load TFLite model and allocate tensors.
    INTERPRETER = tflite.Interpreter(model_path, num_threads=24)
    INTERPRETER.allocate_tensors()

    # Get input and output tensors.
    input_details = INTERPRETER.get_input_details()
    output_details = INTERPRETER.get_output_details()

    # Get input tensor index
    INPUT_LAYER_INDEX = input_details[0]['index']
    OUTPUT_LAYER_INDEX = output_details[0]['index']

def embeddings(sample, model_path):

    global INTERPRETER

    # Does interpreter exist?
    if INTERPRETER == None:
        loadModel(model_path)

    # Reshape input tensor
    INTERPRETER.resize_tensor_input(INPUT_LAYER_INDEX, [len(sample), *sample[0].shape])
    INTERPRETER.allocate_tensors()

    # Extract feature embeddings
    INTERPRETER.set_tensor(INPUT_LAYER_INDEX, np.array(sample, dtype='float32'))
    INTERPRETER.invoke()
    features = INTERPRETER.get_tensor(OUTPUT_LAYER_INDEX)

    return features


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation

def addLinearClassifier(origin_model, num_labels, hidden_units):
    
    # Add dense layer with 'relu' activation if hidden_units > 0
    # The output of the origin_model is used as input for this layer
    if hidden_units > 0:
        x = Dense(hidden_units, activation='relu')(origin_model.output)
    else:
        x = origin_model.output
    
    # Add dense layer outputting raw scores for each class
    predictions = Dense(num_labels)(x)
    
    # Transform raw scores to probabilities with activation
    output = Activation('softmax')(predictions)

    # Now we create the new model
    new_model = Model(inputs=origin_model.input, outputs=output)

    return new_model



def trainNewModel(new_model, train_layers_num, x_train, y_train, x_val, y_val, file_paths_train, file_paths_val, epochs, batch_size, learning_rate, on_epoch_end=None):
    
    tf.keras.backend.clear_session()
    
    # import keras
    from tensorflow import keras

    class FunctionCallback(keras.callbacks.Callback):
        def __init__(self, on_epoch_end=None) -> None:
            super().__init__()
            self.on_epoch_end_fn = on_epoch_end
        
        def on_epoch_end(self, epoch, logs=None):
            if self.on_epoch_end_fn:
                self.on_epoch_end_fn(epoch, logs)
    
    # freeze layer except for train_layers_num
    for layer in new_model.layers[:-train_layers_num]:
        layer.trainable = False
    
    print("Length of trainable Weightslen ", len(new_model.trainable_weights))
    
    # Set random seed
    np.random.seed(42)

    # Shuffle train data
    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]
    file_paths_train = file_paths_train[idx]
    
    # Shuffle validation data
    idx = np.arange(x_val.shape[0])
    np.random.shuffle(idx)
    x_val = x_val[idx]
    y_val = y_val[idx]
    file_paths_val = file_paths_val[idx]

    # Early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        FunctionCallback(on_epoch_end=on_epoch_end)
    ]
    
    # Cosine annealing lr schedule
    lr_schedule = keras.experimental.CosineDecay(learning_rate, epochs * x_train.shape[0] / batch_size)

    # Compile model
    new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0, clipvalue=0.5), 
                       loss='binary_crossentropy', 
                       metrics=['accuracy', 
                                tf.keras.metrics.Precision(name='prec'), 
                                tf.keras.metrics.Recall(name='recall')])
    
    # Creating generators for raw audio input
    train_gen = AudioDataGenerator(file_paths_train, y_train, batch_size=batch_size)
    val_gen = AudioDataGenerator(file_paths_val, y_val, batch_size=batch_size)

    # Train model
    history = new_model.fit(train_gen, 
                              epochs=epochs, 
                              validation_data=val_gen, 
                              callbacks=callbacks)

    return new_model, history


class AudioDataGenerator(tf.keras.utils.Sequence):
#Use AudioDataGenerator for audiofile batch loading

    #Initialize AudioDataGEnerator
    def __init__(self, file_paths, labels, batch_size, samplerate=48000, clip_length=3.0):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.rate = samplerate
        self.clip_length = clip_length

    #Numer of batches per epoch    
    def __len__(self):
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    #generate databatch
    def __getitem__(self, idx):
        batch_x = self.file_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([self.load_audio_file(file_name) for file_name in batch_x]), np.array(batch_y)

    #load the audiofile, extract segment
    def load_audio_file(self, file_path):
        import audio
        import os

        if not os.path.isfile(file_path):
            raise ValueError(f"{file_path} is not a valid file.")
    
        sig, rate = audio.openAudioFile(file_path)
        sig = audio.cropCenter(sig, self.rate, self.clip_length)
        
        # DEBUG        
        # ("Loaded audio file, signal shape: ", sig.shape)
    
        return sig
    
    
    
def debug_model(model, file_paths_train, y_train, batch_size):
        
    # Creating generators for raw audio input
    train_gen = AudioDataGenerator(file_paths_train, y_train, batch_size=batch_size) 

    # mini model just until third layer
    mini_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[2].output)

    # intialize list for ID of batch where NaN might happen
    nan_samples = []

    # Going through all batches
    for batch_index in range(len(train_gen)):
        batch_x, batch_y = train_gen[batch_index]
        
        # Inference with batch
        out = mini_model.predict(batch_x)
        
        # Check for NaN value in batch
        if np.isnan(out).any():
            print(f"NaN values found in batch {batch_index}")

            # Inference for every file in current batch
            for i in range(batch_x.shape[0]):
                sample_out = mini_model.predict(batch_x[i:i+1])

                # Check for NaN value for each file
                if np.isnan(sample_out).any():
                        print(f"NaN values found in sample {i} within batch {batch_index}")
                        print(f"File path: {file_paths_train[batch_index * train_gen.batch_size + i]}")
                        nan_samples.append((batch_index, i))
                    
    print(f"Total samples with NaN values: {len(nan_samples)}")

    return nan_samples

# Define custom Layer LinearSpecLayer
import tensorflow as tf
from tensorflow.keras import layers

class LinearSpecLayer(layers.Layer):
    def __init__(self, sample_rate=48000, spec_shape=(64, 384), frame_step=374, frame_length=512, fmin=250,
                 fmax=15000, data_format='channels_last', **kwargs):
        super(LinearSpecLayer, self).__init__(**kwargs)
        self.sample_rate = sample_rate
        self.spec_shape = spec_shape
        self.frame_step = frame_step
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax
        self.data_format = data_format

    def build(self, input_shape):
        self.mag_scale = self.add_weight(name='magnitude_scaling',
                                         initializer=tf.keras.initializers.Constant(value=1.23),
                                         trainable=True)
        super(LinearSpecLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        # Normalize values between 0 and 1
        inputs = tf.math.subtract(inputs, tf.math.reduce_min(inputs, axis=1, keepdims=True))
        inputs = tf.math.divide(inputs, tf.math.reduce_max(inputs, axis=1, keepdims=True) + 0.000001)
        spec = tf.signal.stft(inputs,
                              self.frame_length,
                              self.frame_step,
                              window_fn=tf.signal.hann_window,
                              pad_end=False,
                              name='stft')

        # magnitude of the complex number
        spec = tf.abs(spec)

        # Only keep bottom half of spectrum
        spec = spec[:, :, :self.frame_length // 4]

        # Convert to power spectrogram
        spec = tf.math.pow(spec, 2.0)

        # Convert magnitudes using nonlinearity
        spec = tf.math.pow(spec, 1.0 / (1.0 + tf.math.exp(self.mag_scale)))

        # Swap axes to fit input shape
        spec = tf.transpose(spec, [0, 2, 1])

        # Add channel axis
        if self.data_format == 'channels_last':
            spec = tf.expand_dims(spec, -1)
        else:
            spec = tf.expand_dims(spec, 1)
# debug
        print(f"final spec shape:{spec}")
# ---
        return spec

    def get_config(self):
        config = super(LinearSpecLayer, self).get_config()
        config.update({
            'sample_rate': self.sample_rate,
            'spec_shape': self.spec_shape,
            'frame_step': self.frame_step,
            'frame_length': self.frame_length,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'data_format': self.data_format
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# Define custom Layer Logmeanexp 
import tensorflow as tf
from tensorflow.keras import layers

def logmeanexp(x, axis=None, keepdims=False, sharpness=1.0):
    xmax = tf.math.reduce_max(x, axis=axis, keepdims=True)
    x = sharpness * (x - xmax)
    y = tf.math.log(tf.math.reduce_mean(tf.exp(x), axis=axis, keepdims=keepdims))
    y = y / sharpness + xmax
    return y

class GlobalLogExpPooling2D(layers.Layer):
    def __init__(self, data_format=None,  **kwargs):
        super(GlobalLogExpPooling2D, self).__init__(**kwargs)
        self.data_format = data_format

    def build(self, input_shape):
        self.sharpness = self.add_weight(name='sharpness', 
                                         shape=(1,), 
                                         initializer=tf.initializers.Constant(value=2.0), 
                                         trainable=True)
        super(GlobalLogExpPooling2D, self).build(input_shape) 

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[3])

    def call(self, x):
        pooled = logmeanexp(x, axis=[1, 2], sharpness=self.sharpness)
        return tf.reshape(pooled, [-1, 420])
    
    def get_config(self):
        config = super(GlobalLogExpPooling2D, self).get_config()
        config.update({'data_format': self.data_format})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)