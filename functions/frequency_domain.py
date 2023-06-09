import scipy.signal as signal
import scipy.fft as fft
from scipy.stats import gmean
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from functions.time_domain import read_wave, split_to_frames

def transform_frames_to_frequency_domain(frames, frame_rate,N_, window_type=None):
    # there are many available windowing functions in scipy.signal.windows, some of which are:
    # 'boxcar', 'triang', 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris' etc...
    
    
    if len(frames.shape) == 1:
        # a single frame
        if window_type is not None:
            # Apply windowing function if specified
            window_func = signal.windows.get_window(window_type, N_)
            windowed_frame = frames * window_func
        else:
            windowed_frame = frames
        return fft.fft(frames)
    
    fft_frames = []
    for frame in frames:
        if window_type is not None:
            # Apply windowing function if specified
            window_func = signal.windows.get_window(window_type, N_)
            windowed_frame = frame * window_func
        else:
            windowed_frame = frame
        fft_frames.append(fft.fft(windowed_frame))
    return np.array(fft_frames)




def visualise_frames(fft_frames, frame_rate,n_, N_, fig=None, subplot_row=1, subplot_col=1):
    magnitude_spectrum = np.abs(fft_frames)
    freq_axis = np.fft.fftfreq(N_, d=1/frame_rate)[:N_//2]

    # Create a subplot for each frame
    if fig is None:
        fig = go.Figure()
        for i in range(n_):
            fig.add_trace(go.Scatter(x=freq_axis, y=magnitude_spectrum[i][:N_//2],
                                    mode='lines', name='Frame {}'.format(i)))

        # Set plot layout
        fig.update_layout(
            title='Frames in Frequency Domain',
            xaxis_title='Frequency (Hz)',
            yaxis_title='Magnitude'
        )

        # Show plot
        fig.show()
    else:
        for i in range(n_):
            fig.add_trace(go.Scatter(x=freq_axis, y=magnitude_spectrum[i][:N_//2],
                                    mode='lines', name='Frame {}'.format(i)),
                          row=subplot_row, col=subplot_col
            )




def compute_volume(fft_frame,N_, in_db=False, spl=False ):
    volume = np.sum(np.abs(fft_frame)**2)/N_
    if in_db: # decibels
        volume = 10*np.log10(volume + 1e-10)
    if spl: #sound pressure level (SPL)
        pref = 2e-5
        volume = 10*np.log10(volume/pref + 1e-10)
    return volume


def visualise_volume(fft_frames,n_, N_, in_db=False, spl=True, fig=None, subplot_row=1, subplot_col=1):
    volumes = np.apply_along_axis(compute_volume,1, fft_frames, N_=N_, in_db=in_db, spl=spl)
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=volumes,
            mode='lines',
            name='Volume',
            line=dict(color='brown')
        ))
        fig.update_layout(
            title='Volume of Each Frame',
            xaxis_title='Frame Index',
            yaxis_title='Volume (dB)',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=volumes, mode='lines', name='Volume',line=dict(color='brown')),
            row=subplot_row, col=subplot_col
        )
        

def compute_frequency_centroid(magnitude, freq_axis, N_):
    '''
    also called spectral centroid
    np.corrcoef(frequency_centroids,zcr)
    array([[1.        , 0.80297413],
        [0.80297413, 1.        ]])
        
    wyszła duża korelacja z zcr tak jak powinno być, więc może jest dobrze
    '''
    mag = magnitude[:N_//2]
    return np.sum(mag*freq_axis)/np.sum(mag)



def visualise_frequency_centroids(fft_frames, frame_rate, N_, n_, fig=None, subplot_row=1, subplot_col=1):
    magnitude_spectrum = np.abs(fft_frames)
    freq_axis = np.fft.fftfreq(N_, d=1/frame_rate)[:N_//2]
    frequency_centroids = np.apply_along_axis(compute_frequency_centroid,1, magnitude_spectrum, freq_axis=freq_axis, N_=N_)
    
    
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=frequency_centroids,
            mode='lines',
            name='frequency_centroids',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title='Frequency Centroid of each frame',
            xaxis_title='Frame index',
            yaxis_title='Frequency centroid (Hz)',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=frequency_centroids, mode='lines', name='frequency_centroids',line=dict(color='orange')),
            row=subplot_row, col=subplot_col
        )




def compute_effective_bandwidth(magnitude, freq_axis, N_):
    # also called spectral spread
    frequency_centroid = compute_frequency_centroid(magnitude, freq_axis, N_)
    mag = magnitude[:N_//2]
    return np.sqrt(np.sum(mag*(freq_axis-frequency_centroid)**2)/np.sum(mag))


def visualise_effective_bandwidths(fft_frames, frame_rate, N_, n_, fig=None, subplot_row=1, subplot_col=1):
    magnitude_spectrum = np.abs(fft_frames)
    freq_axis = np.fft.fftfreq(N_, d=1/frame_rate)[:N_//2]
    effective_bandwidths = np.apply_along_axis(compute_effective_bandwidth,1, magnitude_spectrum, freq_axis=freq_axis, N_=N_)
    
    
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=effective_bandwidths,
            mode='lines',
            name='effective_bandwidths',
            line=dict(color='cyan')
        ))
        fig.update_layout(
            title='Effective Bandwidth of each frame',
            xaxis_title='Frame index',
            yaxis_title='Spread (Hz)',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=effective_bandwidths, mode='lines', name='effective_bandwidths',line=dict(color='cyan')),
            row=subplot_row, col=subplot_col
        )


def compute_spectral_flatness(magnitude,N_):
    mag = magnitude[:N_//2]
    return gmean(mag)/np.mean(mag)




def visualise_spectral_flatness(fft_frames, frame_rate, N_, n_, fig=None, subplot_row=1, subplot_col=1):
    magnitude_spectrum = np.abs(fft_frames)    
    spectral_flatness_vector = np.apply_along_axis(compute_spectral_flatness,1, magnitude_spectrum,N_)
    
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=spectral_flatness_vector,
            mode='lines',
            name='effective_bandwidths',
            line=dict(color='yellow')
        ))
        fig.update_layout(
            title='Spectral Flatness of each frame',
            xaxis_title='Frame index',
            yaxis_title='Flatness',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=spectral_flatness_vector, mode='lines', name='spectral_flatness',line=dict(color='yellow')),
            row=subplot_row, col=subplot_col
        )




def compute_spectral_crest(magnitude,N_):
    mag = magnitude[:N_//2]
    return np.max(mag)/np.sum(mag)



def visualise_spectral_crest(fft_frames, frame_rate, N_, n_, fig=None, subplot_row=1, subplot_col=1):
    magnitude_spectrum = np.abs(fft_frames)    
    spectral_crests = np.apply_along_axis(compute_spectral_crest,1, magnitude_spectrum,N_)
    
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=spectral_crests,
            mode='lines',
            name='spectral_crests',
            line=dict(color='purple')
        ))
        fig.update_layout(
            title='Spectral Crest of each frame',
            xaxis_title='Frame index',
            yaxis_title='Crest',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=spectral_crests, mode='lines', name='spectral_crests',line=dict(color='purple')),
            row=subplot_row, col=subplot_col
        )


def compute_band_energy_ratio(frame_magnitude, freq_axis, N_):
    band_1_2_sep = 630
    band_2_3_sep = 1720
    band_3_4_sep = 4400
    mag = frame_magnitude[:N_//2]
    band_1_energy = np.sum(mag[freq_axis<band_1_2_sep]**2)
    band_2_energy = np.sum(mag[(freq_axis>=band_1_2_sep) & (freq_axis<band_2_3_sep)]**2)
    band_3_energy = np.sum(mag[(freq_axis>=band_2_3_sep) & (freq_axis<band_3_4_sep)]**2)
    full_energy = np.sum(mag**2)
    return band_1_energy/full_energy, band_2_energy/full_energy, band_3_energy/full_energy
    
    
    
def visualise_band_energy_ratio(fft_frames, frame_rate, N_, n_, fig=None, subplot_row=1, subplot_col=1):
    magnitude_spectrum = np.abs(fft_frames)
    freq_axis = np.fft.fftfreq(N_, d=1/frame_rate)[:N_//2]
    ratios_matrix = np.apply_along_axis(compute_band_energy_ratio,1, magnitude_spectrum, freq_axis=freq_axis, N_=N_)
    ratio_1 = ratios_matrix[:,0]
    ratio_2 = ratios_matrix[:,1]
    ratio_3 = ratios_matrix[:,2]
    
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=ratio_1,
            mode='lines',
            name='ERSB1',
            line=dict(color='green')
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=ratio_2,
            mode='lines',
            name='ERSB2',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=ratio_3,
            mode='lines',
            name='ERSB3',
            line=dict(color='red')
        ))
        
        

        fig.update_layout(
            title='Band Energy Ratio',
            xaxis_title='Frame index',
            yaxis_title='Ratio',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=ratio_1, mode='lines', name='ERSB1',line=dict(color='green')),
            row=subplot_row, col=subplot_col
        )
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=ratio_2, mode='lines', name='ERSB2',line=dict(color='blue')),
            row=subplot_row, col=subplot_col
        )
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=ratio_3, mode='lines', name='ERSB3',line=dict(color='red')),
            row=subplot_row, col=subplot_col
        )


def compute_base_frequency(frame, sampling_frequency, min_freq=50, max_freq=400):
    real_cepstrum = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(frame)))))
    min_quefrency = int(sampling_frequency / max_freq)
    max_quefrency = int(sampling_frequency / min_freq)
    local_max_quefrency = np.argmax(real_cepstrum[min_quefrency:max_quefrency]) + min_quefrency
    base_frequency = sampling_frequency / local_max_quefrency

    return real_cepstrum, base_frequency

def visualise_base_frequency(frames, frame_rate, N_, n_, fig=None, subplot_row=1, subplot_col=1):
    base_frequencies = []
    for frame in frames:
        _, base_frequency = compute_base_frequency(frame, frame_rate)
        base_frequencies.append(base_frequency)
    base_frequencies = np.array(base_frequencies)  
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=base_frequencies,
            mode='lines',
            name='base_frequencies',
            line=dict(color='pink')            
        ))
        fig.update_layout(
            title= 'Base frequencies',
            xaxis_title='Frame index',
            yaxis_title='Frequency (Hz)',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=base_frequencies, mode='lines', name='base_frequencies', line=dict(color='pink')),
            row=subplot_row, col=subplot_col
        )


def visualise_all(frames,frame_rate, n_, N_, window_type=None, in_db=False, spl=True, legend=True):
    fft_frames = transform_frames_to_frequency_domain(frames, frame_rate, N_, window_type=window_type)
    fig = make_subplots(rows=8, cols=1, vertical_spacing=0.05,
                        subplot_titles=("Frames in Frequency Domain", "Volume", "Frequency Centroids", "Effective Bandwidths", "Spectral Flatness Measure","Spectral Crest Factor","Band Energy Ratio","Base Frequencies"))
    
    fig.update_layout(height=2200,showlegend=legend)
    
    visualise_frames(fft_frames, frame_rate, n_, N_, fig=fig, subplot_row=1, subplot_col=1)
    visualise_volume(fft_frames,n_, N_, in_db=in_db, spl=spl, fig=fig, subplot_row=2, subplot_col=1)
    visualise_frequency_centroids(fft_frames, frame_rate, N_, n_, fig=fig, subplot_row=3, subplot_col=1)
    visualise_effective_bandwidths(fft_frames, frame_rate, N_, n_, fig=fig, subplot_row=4, subplot_col=1)   
    visualise_spectral_flatness(fft_frames, frame_rate, N_, n_, fig=fig, subplot_row=5, subplot_col=1)
    visualise_spectral_crest(fft_frames, frame_rate, N_, n_, fig=fig, subplot_row=6, subplot_col=1)
    visualise_band_energy_ratio(fft_frames, frame_rate, N_, n_, fig=fig, subplot_row=7, subplot_col=1)
    visualise_base_frequency(frames, frame_rate, N_, n_, fig=fig, subplot_row=8, subplot_col=1)

    fig.layout.xaxis.update(title="Frequency (Hz)")
    fig.layout.yaxis.update(title="Magnitude")
    
    fig.layout.xaxis2.update(title="Frame index")
    if in_db or spl:
        fig.layout.yaxis2.update(title="Volume (dB)")
        
    fig.layout.yaxis3.update(title="Frequency centroid (Hz)")
    fig.layout.xaxis3.update(title="Frame index")
    
    fig.layout.yaxis4.update(title="Spread (Hz)")
    fig.layout.xaxis4.update(title="Frame index")
    
    fig.layout.yaxis5.update(title="Flatness")
    fig.layout.xaxis5.update(title="Frame index")   
        
    fig.layout.yaxis6.update(title="Crest")
    fig.layout.xaxis6.update(title="Frame index")   
    
    
    fig.layout.yaxis7.update(title="Ratio")
    fig.layout.xaxis7.update(title="Frame index")   

    fig.layout.yaxis8.update(title="Frequency (Hz)")
    fig.layout.xaxis8.update(title="Frame index")
    fig.show()
            
        
        
def visualise_spectrogram(path,percent_frame_size,percent_hop_length, window_type='hann',figsize=None,time_in_frames=False,log_amplitude=False):
    # it's not a mel spectrogram
    audio, frame_rate, audio_time, n_samples  = read_wave(path,display=False)
    frames, n_, N_= split_to_frames(audio, frame_rate, percent_frame_size=percent_frame_size, percent_hop_length=percent_hop_length)
    fft_frames = transform_frames_to_frequency_domain(frames, frame_rate, N_, window_type='hann')
    magnitude_spectrum = np.abs(fft_frames)[:, :N_//2]
    if log_amplitude:
        magnitude_spectrum = 10*np.log10(magnitude_spectrum.T)
    if figsize is None:
        figsize = (15,10)
    plt.figure(figsize=figsize)
    plt_mag = plt.imshow(magnitude_spectrum, origin='lower',aspect='auto')
    plt.yticks(np.round(np.linspace(0, N_//2, 10),3), np.array(np.round(np.linspace(0, frame_rate/2, 10),2),dtype=int))
    plt.ylabel('Frequency [Hz]')
    if time_in_frames:
        plt.xlabel('Frames')
    else:
        plt.xlabel('Time [s]')
        plt.xlim(0, audio_time)
        labels = np.linspace(0, audio_time, 10)
        str_labels = [f'{num:.1f}' for num in np.around(labels, decimals=1)]
        plt.xticks(np.linspace(0,n_-1,10),labels= str_labels)
    plt.colorbar()
    plt.show()
    



# modified to show fundamental frequency
# 
# def compute_base_frequency(signal, sampling_frequency, min_freq=50, max_freq=400):
#     real_cepstrum = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(signal)))))
#     min_quefrency = int(sampling_frequency / max_freq)
#     max_quefrency = int(sampling_frequency / min_freq)
#     local_max_quefrency = np.argmax(real_cepstrum[min_quefrency:max_quefrency]) + min_quefrency
#     base_frequency = sampling_frequency / local_max_quefrency

#     return real_cepstrum, base_frequency


# def visualise_ceps(audio, frame_rate, min_freq=50, max_freq=400, fig=None, subplot_row=1, subplot_col=1):
#     ceps, base_freq = compute_base_frequency(audio, frame_rate)
#     actual_ceps = ceps[min_freq:max_freq]
#     if fig is None:
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(
#             x=np.arange(min_freq, max_freq - min_freq),
#             y=actual_ceps,
#             mode='lines',
#             name='Cepstrum'
#         ))
#         fig.update_layout(
#             title='Cepstrum of Each Sample',
#             xaxis_title='Sample Index',
#             yaxis_title='Cepstrum',
#         )
#         fig.show()
#     else:
#         fig.add_trace(
#             go.Scatter(x=np.arange(min_freq, max_freq - min_freq), y=actual_ceps, mode='lines', name='Cepstrum'),
#             row=subplot_row, col=subplot_col
#         )
#     print(f"Fundamental frequency: {base_freq}")
    
    
