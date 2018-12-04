# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import scipy.io.wavfile as wav
import numpy as np
import matplotlib.pylab as plt
from cls_psychoacoustic_model import PsychoacousticModel as psm
from kapre import time_frequency as tf
from keras.models import Sequential


def stft_keras(n_dft, n_hop):
    model = Sequential()
    model.add(tf.Spectrogram(n_dft=n_dft, n_hop=n_hop, padding='same',
              power_spectrogram=1.0, return_decibel_spectrogram=False,
              trainable_kernel=False))
    model.compile('adam', 'mse')
    return model


def main():
    n_dft = 2048
    n_hop = 512
    int_val = 32768
    # Waveform
    fs, audio = wav.read('wav/dog_bark.wav')
    if audio.shape[1] == 2:
        audio = np.mean(audio, axis=1, keepdims=True).T
        audio = np.expand_dims(audio, 0)
        audio /= int_val

    # Keras-based STFT
    model = stft_keras(n_dft, n_hop)
    out_spect = model.predict(audio)[0, :, :, 0].T

    # Psychoacoustics
    my_psm = psm(N=2048, fs=fs, nfilts=32, type='rasta',
                 width=1.0, minfreq=0, maxfreq=22050)

    mt = my_psm.maskingThreshold(out_spect)

    plt.figure()
    plt.imshow(out_spect.T, aspect='auto', origin='lower')
    plt.figure()
    plt.imshow(mt.T, aspect='auto', origin='lower')
    plt.show(block=False)


if __name__ == "__main__":
    main()

# EOF
