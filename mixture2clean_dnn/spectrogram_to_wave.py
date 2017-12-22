"""
Summary:  Recover spectrogram to wave. 
Author:   Qiuqiang Kong
Created:  2017.09
Modified: -
"""
import numpy as np
import numpy
import decimal
    
def recover_wav(pd_abs_x, gt_x, n_overlap, winfunc, wav_len=None):
    """Recover wave from spectrogram. 
    If you are using scipy.signal.spectrogram, you may need to multipy a scaler
    to the recovered audio after using this function. For example, 
    recover_scaler = np.sqrt((ham_win**2).sum())
    
    Args:
      pd_abs_x: 2d array, (n_time, n_freq)
      gt_x: 2d complex array, (n_time, n_freq)
      n_overlap: integar. 
      winfunc: func, the analysis window to apply to each frame.
      wav_len: integer. Pad or trunc to wav_len with zero. 
      
    Returns:
      1d array. 
    """
    x = real_to_complex(pd_abs_x, gt_x)
    x = half_to_whole(x)
    frames = ifft_to_wav(x)
    (n_frames, n_window) = frames.shape
    s = deframesig(frames=frames, siglen=0, frame_len=n_window, 
                   frame_step=n_window-n_overlap, winfunc=winfunc)
    if wav_len:
        s = pad_or_trunc(s, wav_len)
    return s
    
def real_to_complex(pd_abs_x, gt_x):
    """Recover pred spectrogram's phase from ground truth's phase. 
    
    Args:
      pd_abs_x: 2d array, (n_time, n_freq)
      gt_x: 2d complex array, (n_time, n_freq)
      
    Returns:
      2d complex array, (n_time, n_freq)
    """
    theta = np.angle(gt_x)
    cmplx = pd_abs_x * np.exp(1j * theta)
    return cmplx
    
def half_to_whole(x):
    """Recover whole spectrogram from half spectrogram. 
    """
    return np.concatenate((x, np.fliplr(np.conj(x[:, 1:-1]))), axis=1)

def ifft_to_wav(x):
    """Recover wav from whole spectrogram"""
    return np.real(np.fft.ifft(x))

def pad_or_trunc(s, wav_len):
    if len(s) >= wav_len:
        s = s[0 : wav_len]
    else:
        s = np.concatenate((s, np.zeros(wav_len - len(s))))
    return s

def recover_gt_wav(x, n_overlap, winfunc, wav_len=None):
    """Recover ground truth wav. 
    """
    x = half_to_whole(x)
    frames = ifft_to_wav(x)
    (n_frames, n_window) = frames.shape
    s = deframesig(frames=frames, siglen=0, frame_len=n_window, 
                   frame_step=n_window-n_overlap, winfunc=winfunc)
    if wav_len:
        s = pad_or_trunc(s, wav_len)
    return s

def deframesig(frames,siglen,frame_len,frame_step,winfunc=lambda x:numpy.ones((x,))):    
    """Does overlap-add procedure to undo the action of framesig.
    Ref: From https://github.com/jameslyons/python_speech_features
    
    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(frames)[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'
 
    indices = numpy.tile(numpy.arange(0,frame_len),(numframes,1)) + numpy.tile(numpy.arange(0,numframes*frame_step,frame_step),(frame_len,1)).T
    indices = numpy.array(indices,dtype=numpy.int32)
    padlen = (numframes-1)*frame_step + frame_len   
    
    if siglen <= 0: siglen = padlen
    
    rec_signal = numpy.zeros((padlen,))
    window_correction = numpy.zeros((padlen,))
    win = winfunc(frame_len)
    
    for i in range(0,numframes):
        window_correction[indices[i,:]] = window_correction[indices[i,:]] + win + 1e-15 #add a little bit so it is never zero
        rec_signal[indices[i,:]] = rec_signal[indices[i,:]] + frames[i,:]
        
    rec_signal = rec_signal/window_correction
    return rec_signal[0:siglen]
    
def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))