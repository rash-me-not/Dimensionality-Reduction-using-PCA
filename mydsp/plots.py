'''
plots - plot graph for spectrograms
'''
from mydsp.dftstream import DFTStream
from mydsp.multifileaudioframes import MultiFileAudioFrames
import numpy as np


def spectrogram(filenames, frame_adv, frame_len):
    multifiles = MultiFileAudioFrames(filenames,frame_adv,frame_len)

    #Computing time axis



    multifile_frame = []
    for frame in multifiles:
        multifile_frame.append(frame)

    dftStream = DFTStream(multifiles)

    frames = len(multifiles.samplefile)
    tidx = int(len(multifiles.samplefile.data) / multifiles.Fs)
    time = np.linspace(0, tidx, frames)

    dft_intensity = []
    for frame in dftStream:
        dft_intensity.append(frame)

    freq = dftStream.get_Hz()

    return  time, dft_intensity,freq


