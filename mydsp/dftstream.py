'''
dftstream - Streamer for Fourier transformed spectra
'''

import numpy as np
import scipy.signal as signal


class DFTStream:

    def __init__(self, frame_stream, specfmt="dB"):
        '''
           DFTStream(frame_stream, specfmt)
           Create a stream of discrete Fourier transform (DFT) frames using the
           specified sample frame stream. Only bins up to the Nyquist rate are
           returned in the stream Optional arguments:

           specfmt - DFT output:
               "complex" - return complex DFT results
                "dB" [default] - return power spectrum 20log10(magnitude)
                "mag^2" - magnitude squared spectrum
        '''

        self.frame_stream = frame_stream
        self.specfmt = specfmt
        self.Nyquist_rate = frame_stream.get_Nyquist()
        self.Fs = frame_stream.get_Fs
        self.frame_len = frame_stream.get_framelen_samples()
        self.frame_Nyquist = int(frame_stream.get_framelen_samples() / 2)

    def shape(self):
        "shape() - Return dimensions of tensor yielded by next()"
        return np.asarray([self.frame_stream.get_framelen_samples, 1])

    def size(self):
        "size() - number of elements in tensor generated by iterator"
        return np.asarray(np.product(self.shape()))


    def get_Hz(self):
        """get_Hz() - Return list of frequencies associated with each
        spectral bin.  (List is of the same size as the # of spectral
        bins up to the Nyquist rate, or half the frame lenght)
        """
        return np.arange(self.frame_Nyquist) / self.frame_Nyquist * self.Nyquist_rate


    def __iter__(self):
        "iter() Return iterator for stream"
        self.frame_iter = iter(self.frame_stream)
        return self


    def __next__(self):
        "next() Return next DFT frame"
        nextFrame = next(self.frame_iter)
        window = signal.get_window("hamming", self.frame_len)
        windowed_x = nextFrame * window
        X = np.fft.fft(windowed_x)

        self.magX = np.abs(X)
        self.mag_dB = 20 * np.log10(self.magX)

        return self.mag_dB[0:self.frame_Nyquist]


    def __len__(self):
        "len() - Number of tensors in stream"
        return len(self.frame_stream)