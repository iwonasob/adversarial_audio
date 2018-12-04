import keras
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Permute
from keras.utils import np_utils
import keras.backend as K
import tensorflow as tf

from kapre.time_frequency import Melspectrogram, Spectrogram
from cls_psychoacoustic_model import PsychoacousticModel as psm
import numpy as np
import os
import scipy.io.wavfile as wav
import argparse

#AUDIO ANALYSIS PARAMETERS
N_FFT = 1024         # number of FFT bins
HOP_SIZE=1024        # number of samples between consecutive windows of STFT
SR=44100             # sampling frequency 
WIN_SIZE = 1024      # number of samples in each STFT window
WINDOW_TYPE = 'hann' # the windowin function
FEATURE= 'mel'       # feature representation

# Mel band parameters
N_MELS = 40          # number of mel bands

MAX_LENGTH_S=4       # maximum length of a file in seconds
MAX_LENTGH_SAMP=int(np.ceil(MAX_LENGTH_S*SR/WIN_SIZE)) # corresponding maximum length in samples

#workspace = '/user/cvsspstf/is0017/adversarial_audio'
workspace = '/home/mis/Documents/Python/Projects/dist/adversarial_audio'
modelpath = os.path.join(workspace, 'models')
keras_modelname = 'crnn.hdf5'
keras_modelpath = os.path.join(modelpath, keras_modelname)


# Psychoacoustic model
my_psm = psm(N=N_FFT, fs=SR, nfilts=32, type='rasta',
             width=1.0, minfreq=0, maxfreq=22050)



def custom_loss(input_data , class_idx, out, audio_original):
    y_true = np_utils.to_categorical(class_idx, num_classes=10)
    cross_entropy = K.categorical_crossentropy(y_true, out)
    distortion = K.sum((input_data - audio_original)**2, axis=1, keepdims=False) 
    total_loss = cross_entropy + 0.5 *distortion
    return total_loss


### build combined model
def build_keras_model(n_dft, n_hop, SR, MAX_LENGTH_S):
    # load model to hack:
    cnn_model = keras.models.load_model(keras_modelpath)

    input_shape = (1, SR * MAX_LENGTH_S)

    wav_input = Input(shape=input_shape, name='input_wav')

    mag_spec = Spectrogram(n_dft=n_dft, n_hop=n_hop, padding='same')(wav_input)

    mag_model = Model(inputs=wav_input, outputs=mag_spec, name='mag_model')
    mag_model.summary()

    mel_output = Melspectrogram(n_dft=N_FFT, n_hop=HOP_SIZE, input_shape=input_shape,
                                padding='same', sr=SR, n_mels=N_MELS,
                                fmin=0.0, fmax=SR / 2, power_melgram=1.0,
                                return_decibel_melgram=True, trainable_fb=False,
                                trainable_kernel=False,
                                name='trainable_stft')(wav_input)

    mel_model = Model(inputs=wav_input, outputs=mel_output, name='mel_model')
    mel_model.summary()

    # now you can create a global model as follows:
    inputs = Input(shape=input_shape)
    x = mel_model(inputs)
    x = Permute((2, 1, 3))(x)
    predictions = cnn_model(x)
    full_model = Model(input=inputs, output=predictions)

    # Verify things look as expected
    full_model.summary()

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    full_model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=5e-5, momentum=0.9),
        metrics=['accuracy'])

    return mag_model, full_model


def modify_audio(audio, model, mag_model, class_idx):
    
    # Classification
    audio_original = audio

    input_data = model.input

    out = model.output

    # Regression
    out_spect = mag_model.predict(audio)[0, :, :, 0].T
    mt = my_psm.maskingThreshold(out_spect).T[None, :, :, None]
    imt = (1./(mt + 1e-4) ) ** 2.

    loss = custom_loss(input_data, class_idx, out, audio_original)

    
    grads_history=[]
    grads = 0
    grads = K.gradients(loss, input_data)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    iterate = K.function([input_data], [loss, grads])
    
    for i in range(500):
        loss_value, grads_value = iterate([audio])
        grads_history.append(grads_value)
        audio_tmp = audio - 0.1 * grads_value
        audio = audio_tmp
        #print('Current loss value:', loss_value)
        #print('Current grads value:', grads_value)

    return np.squeeze(audio[0][0])
    
### load wavfile and predict
def main():
    """
    Do the attack here.
    """
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--in', type=str, dest="input", nargs='+',
                        required=True,
                        help="Input audio .wav file(s), at 44100KHz (separated by spaces)")
    parser.add_argument('--class', type=int, dest="class_idx", nargs='+',
                        required=True,
                        help="Index of the class we want to achieve")
    args = parser.parse_args()

    fs, audio = wav.read(args.input[0])
    assert fs == 44100
    if audio.shape[1] == 2:
        audio = audio[:,0]
        
    class_idx = args.class_idx[0]
    adv_wav = os.path.join(workspace,  "adv_"+str(class_idx)+".wav")    
        
    audio = audio[:SR*MAX_LENGTH_S]
    audio = audio[np.newaxis,np.newaxis,:]

    mag_model, full_model = build_keras_model(N_FFT, HOP_SIZE, SR, MAX_LENGTH_S)

    prediction = full_model.predict(audio)
    print(prediction)
    print(np.argmax(prediction))
    
    adv_audio = modify_audio(audio, full_model, mag_model, class_idx)
    
    wav.write(adv_wav, fs, adv_audio.astype(np.int16))
    adv_audio = adv_audio[np.newaxis,np.newaxis,:]
    adv_prediction = full_model.predict(adv_audio, steps=1)
    print(adv_prediction)
    print(np.argmax(adv_prediction))
    
main()