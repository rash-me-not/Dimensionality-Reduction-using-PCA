'''
driver assignment 2
Name:  
'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import scale
from mydsp import multifileaudioframes
import mydsp.plots as plotSpectrogram
import mydsp.utils as utils
from mydsp.dftstream import DFTStream
import mydsp.pca as pca


def plot_narrowband_wideband(filename):
    """plot_narrowband_widband(filename)
    Given a speech file, display narrowband and wideband
    spectrograms of the speech.
    """

    #Computing time to plot on x-axis

    plt.figure()
    plt.subplot(2,1,1)
    time , intensity_narrow,freq = plotSpectrogram.spectrogram([filename], 20, 40)
    plt.annotate('Narrowband spectrogram has poor time resolution', xy=(1.55, 1000), xycoords='data',
                xytext=(0.8, 0.95), color='yellow', textcoords='axes fraction',
                arrowprops=dict(facecolor='white', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )
    plt.title("Narrowband Spectrogram (Frame length = 40 sec)")
    plt.xlabel("Time (in sec)")
    plt.ylabel("Frequency (in Hz)")
    plt.pcolormesh(time,freq,np.asarray(intensity_narrow).transpose())

    plt.subplot(2,1,2)
    time , intensity_wide,freq = plotSpectrogram.spectrogram([filename], 3, 6)
    plt.annotate('Wideband spectrogram has better time resolution', xy=(1.55, 1000), xycoords='data',
                xytext=(0.8, 0.95), color='yellow', textcoords='axes fraction',
                arrowprops=dict(facecolor='white', shrink=0.05),
                horizontalalignment='right', verticalalignment='top',
                )
    plt.pcolormesh(time,freq,np.asarray(intensity_wide).transpose())
    plt.title("Wideband Spectrogram (Frame length = 6 sec)")
    plt.xlabel("Time (in sec)")
    plt.ylabel("Frequency (in Hz)")
    plt.tight_layout()
    plt.show()

def pca_analysis(corpus_dir):
    """pca_analysis(corpus_dir)
    Given a directory containing a speech corpus, compute spectral features
    and conduct a PCA analysis of the data.
    """

    file_list  = utils.get_corpus(corpus_dir)

    multifile = multifileaudioframes.MultiFileAudioFrames(file_list, 10, 20)
    dftstream = DFTStream(multifile)

    frame_list = []
    for i in dftstream:
        frame_list.append(i)

    frame_list = scale(np.asarray(frame_list))
    pca.PCA(frame_list)



def speech_silence(filename):
    """speech_silence(filename)
    Given speech file filename, train a 2 mixture GMM on the
    RMS intensity and label each frame as speech or silence.
    Provides a plot of results.
    """
    pass



if __name__ == '__main__':
    """If invoked as the main module, e.g. python driver.py, execute
    """
    pca_analysis("./ti-digits-train-women/woman")

