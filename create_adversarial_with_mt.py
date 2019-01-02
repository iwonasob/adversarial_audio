import keras
import librosa
from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Permute
from keras.utils import np_utils
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

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
my_psm = psm(N=N_FFT*2, fs=SR, nfilts=32, type='rasta',
             width=1.0, minfreq=0, maxfreq=22050)

def compute_td_mt(x):
    """
        A function that accepts an input waveform and returns another one that contains
        the parts of the input waveform which are perceptually masked.
    """
    # Normalize waveform
    n_f = (2**16)/2.
    dft_nf = np.sqrt(N_FFT*2)
    x_norm = np.float32(x/n_f)

    c_x = librosa.stft(x_norm, n_fft=N_FFT*2, hop_length=WIN_SIZE//2,
                       win_length=WIN_SIZE, window='hann',
                       center=True)
    mag_x = np.abs(c_x).T
    mag_x /= dft_nf

    mt = my_psm.maskingThreshold(mag_x)

    # The masking threshold to auralize
    c_mt = mt.T * dft_nf * np.exp(-1j * np.angle(c_x))

    wv_mt = librosa.istft(c_mt, hop_length=WIN_SIZE//2, win_length=WIN_SIZE, window='boxcar', center=True)

    if len(wv_mt) > len(x):
        wv_mt = wv_mt[:len(x)]

    elif len(x) > len(wv_mt):
        diff = len(x) - len(wv_mt)
        wv_mt = np.hstack((wv_mt, np.ones(diff)))

    wv_mt *= n_f
    wv_mt = np.int16(wv_mt)

    return wv_mt

def custom_loss_with_mt(input_data, class_idx, out, audio_original, mt_signal):
    epsilon = 1e-1
    y_true = np_utils.to_categorical(class_idx, num_classes=10)
    cross_entropy = K.categorical_crossentropy(y_true, out)
    distortion = K.sum(K.pow(input_data - audio_original, 2.) / (K.pow(mt_signal, 2.) + epsilon), axis=2, keepdims=False)
    total_loss = distortion + cross_entropy
    return total_loss


### build combined model
def build_keras_model(SR, MAX_LENGTH_S):
    # load model to hack:
    cnn_model = keras.models.load_model(keras_modelpath)

    input_shape = (1, SR * MAX_LENGTH_S)

    wav_input = Input(shape=input_shape, name='input_wav')

    mel_output = Melspectrogram(n_dft=N_FFT, n_hop=HOP_SIZE, input_shape=input_shape,
                                padding='same', sr=SR, n_mels=N_MELS,
                                fmin=0.0, fmax=SR/2, power_melgram=1.0,
                                return_decibel_melgram=True, trainable_fb=False,
                                trainable_kernel=False,
                                name='trainable_stft')(wav_input)

    mel_model = Model(inputs=wav_input, outputs=mel_output, name='mel_model')
    #mel_model.summary()

    # now you can create a global model as follows:
    inputs = Input(shape=input_shape, name='input_data')
    input_imt = Input(shape=input_shape, name='input_threshold')
    x = mel_model(inputs)
    x = Permute((2, 1, 3))(x)
    predictions = cnn_model(x)
    full_model = Model(input=[inputs, input_imt], output=predictions)

    # Verify things look as expected
    #full_model.summary()

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    full_model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=5e-5, momentum=0.9),
        metrics=['accuracy'])

    return full_model


def modify_audio(audio, mt, model, class_idx):
    
    # Classification
    audio_original = audio
    noisy_audio = audio + 2.*mt

    input_data = model.input[0]
    input_mt = model.input[1]

    out = model.output

    loss = custom_loss_with_mt(input_data, class_idx, out, audio_original, input_mt)

    grads_history = []
    grads = K.gradients(loss, input_data)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    iterate = K.function([input_data, input_mt], [loss, grads])
    
    for i in range(500):
        loss_value, grads_value = iterate([noisy_audio, mt])
        grads_history.append(grads_value)
        audio_tmp = audio - 0.1 * grads_value
        noisy_audio = audio_tmp
        """
        if i % 20 == 0:
            plt.figure(1)
            plt.plot(grads_value[0, 0, :])
            plt.figure(2)
            plt.plot(loss_value)
            plt.show()
        """
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
        audio = audio[:, 0]

    # Compute the masking threshold
    mt = compute_td_mt(audio)

    class_idx = args.class_idx[0]
    adv_wav = os.path.join(workspace,  "adv_"+str(class_idx)+".wav")
    mt_wav = os.path.join(workspace,  "mt_"+str(class_idx)+".wav")

    # Waveform cutting/reshaping
    audio = audio[:SR*MAX_LENGTH_S]
    audio = audio[np.newaxis, np.newaxis, :]
    mt = mt[:SR*MAX_LENGTH_S]
    mt = mt[np.newaxis, np.newaxis, :]

    full_model = build_keras_model(SR, MAX_LENGTH_S)

    prediction = full_model.predict([audio, mt])
    print(np.argmax(prediction))
    
    adv_audio = modify_audio(audio, mt, full_model, class_idx)
    
    wav.write(adv_wav, fs, adv_audio.astype(np.int16))
    wav.write(mt_wav, fs, mt)

    adv_audio = adv_audio[np.newaxis, np.newaxis, :]
    adv_prediction = full_model.predict([adv_audio, mt], steps=1)
    print(np.argmax(adv_prediction))

    
main()