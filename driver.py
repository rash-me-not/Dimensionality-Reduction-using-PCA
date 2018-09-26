'''
driver assignment 2
Name:  
'''

import matplotlib.pyplot as plt
import numpy as np
import mydsp.plots as plotSpectrogram
import mydsp.utils as utils
from mydsp.pca import PCA
import os


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

    dft_intensity, time, freq = plotSpectrogram.spectrogram(file_list,10,20)

    pca = PCA(dft_intensity)

    eigen_values = pca.eig_vals

    variance_captured = 100 * np.cumsum(eigen_values)/sum(eigen_values)
    plt.plot(np.arange(len(eigen_values)), variance_captured)

    plt.xlabel("Components")
    plt.ylabel("% of Variance captured")
    plt.show()

    decile = 10
    for counter, value in enumerate(variance_captured):

        while value >= decile:
            print("{} components required for >={}% ".format(counter+1,decile))

            decile = decile + 10

    filename = os.path.join(corpus_dir,"ac/6a.wav").replace("\\","/")
    dft_intensity_6a, time_6a, freq_6a = plotSpectrogram.spectrogram([filename], 10, 20)

    pca_6a = PCA(dft_intensity_6a)
    eigen_values_6a = pca_6a.eig_vals

    #define the threshold to capture number of elements in the data
    threshold = 80
    threshold_eig_val_6a = []
    sum_cumulative=0
    for i in range(len(eigen_values_6a)):
        sum_cumulative = sum_cumulative + np.sum(eigen_values_6a[i])
        if (sum_cumulative/sum(eigen_values_6a))*100 <= threshold:
            threshold_eig_val_6a.append(eigen_values_6a[i])
        else:
            break;

    dimensions = len(threshold_eig_val_6a)
    eigen_vec_6a = pca_6a.transform(dft_intensity_6a,dimensions)

    plt.figure()
    plt.pcolormesh(time_6a,np.arange(0,dimensions+1),(np.transpose(np.abs(eigen_vec_6a))))
    plt.xlabel("Time (in sec)")
    plt.ylabel("PCA Component")
    plt.show()


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



