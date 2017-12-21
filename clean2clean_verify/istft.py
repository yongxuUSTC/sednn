import numpy as np


class ISTFT():
    """
    Inverse Short-Time Fourier Transform
    Input should be a complex spectrogram, output is a signal vector
    usage:
    istft=ISTFT( window=None, fft_size=n_fft, hop_size=hop_size, sample_rate=sr)
    y=istft.process(Y_STFT)
    """

    def __init__(self, window=None, fft_size=1024, hop_size=512, sample_rate=44100):
        if window is None:
            self.window = np.hanning(fft_size)
        else:
            self.window = window
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate

    def process(self, spectrogram):
        frames = np.fft.irfft(spectrogram.T)
        result_length = ((frames.shape[0] - 1) * self.hop_size) + self.fft_size
        result = np.zeros((result_length, 1))
        for i in range(frames.shape[0]):
            indices = i * self.hop_size + np.r_[0:self.fft_size]
            result[indices] = result[indices] + frames[i, :].reshape(self.fft_size, 1)
        result = result[self.hop_size:result_length - self.hop_size]
        return result