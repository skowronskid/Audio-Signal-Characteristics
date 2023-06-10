import numpy as np
from functions.time_domain import *
from functions.frequency_domain import *

def preemphasis(signal, coeff=0.97, frame_rate=None, display=False):
    preepmhasized_signal = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    if display:
        displayhook(Audio(preepmhasized_signal, rate=frame_rate))
    return preepmhasized_signal



def get_filter_banks(n_filters=16, NFFT=512, frame_rate=None, low_freq_mel=0, high_freq_mel=None):
    if high_freq_mel is None:
        high_freq_mel = (2595 * np.log10(1 + (frame_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)  
    hz_points = (700 * (10**(mel_points / 2595) - 1))  
    bins = np.floor((NFFT + 1) * hz_points / frame_rate)

    fbank = np.zeros((n_filters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, n_filters + 1):
        f_m_minus = int(bins[m - 1])   # left
        f_m = int(bins[m])             # center
        f_m_plus = int(bins[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    return fbank

def get_dct_coefficients(input_signals, M):
    num_signals = len(input_signals)
    dct_coefficients = np.zeros((num_signals, M))  # Initialize the array for DCT coefficients

    for n in range(num_signals):
        input_signal = input_signals[n]

        for i in range(M):
            for m in range(M):
                dct_coefficients[n, i] += np.log(input_signal[m]) * np.cos((np.pi * i) / (2 * M) * (m - 0.5))

        dct_coefficients[n] *= np.sqrt(2 / M)  # Apply the scaling factor

    return dct_coefficients


def get_energy(frame):
    return np.sum(np.square(frame))

def mfcc_pipeline(path):
    percent_frame_size = 0.15
    percent_hop_length = 0.5
    audio, frame_rate, audio_time, n_samples  = read_wave(path, display=False)
    preepmhasized_audio = preemphasis(audio, frame_rate=frame_rate, display=False)
    frames, n_, N_ = split_to_frames(preepmhasized_audio, frame_rate, percent_frame_size=percent_frame_size, percent_hop_length=percent_hop_length, set_frame_size=256)
    fft_frames = transform_frames_to_frequency_domain(frames, frame_rate, N_, window_type='hamming')
    amplitude = np.abs(fft_frames)[:,:N_//2+1]
    magnitude = amplitude**2
    filters = get_filter_banks(n_filters=16, NFFT=256, frame_rate=frame_rate)
    filtered_magnitude = np.dot(magnitude, filters.T)
    mfcc = get_dct_coefficients(filtered_magnitude, filtered_magnitude.shape[1])
    energy = np.apply_along_axis(get_energy, 1, frames)
    
    return mfcc, energy
    