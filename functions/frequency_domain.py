import scipy.signal as signal
import scipy.fft as fft
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns


def transform_frames_to_frequency_domain(frames, frame_rate, N_, window_type=None):
    # there are many available windowing functions in scipy.signal.windows, some of which are: 'boxcar', 'triang',
    # 'blackman', 'hamming', 'hann', 'bartlett', 'flattop', 'parzen', 'bohman', 'blackmanharris' etc...

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


def visualise_frames(fft_frames, frame_rate, n_, N_, fig=None, subplot_row=1, subplot_col=1):
    # Calculate magnitude spectrum
    magnitude_spectrum = np.abs(fft_frames)

    # Generate frequency axis
    freq_axis = np.fft.fftfreq(N_, d=1 / frame_rate)[:N_ // 2]

    # Create a subplot for each frame
    if fig is None:
        fig = go.Figure()
        for i in range(n_):
            fig.add_trace(go.Scatter(x=freq_axis, y=magnitude_spectrum[i][:N_ // 2],
                                     mode='lines', name='Frame {}'.format(i + 1)))

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
            fig.add_trace(go.Scatter(x=freq_axis, y=magnitude_spectrum[i][:N_ // 2],
                                     mode='lines', name='Frame {}'.format(i + 1)),
                          row=subplot_row, col=subplot_col
                          )


def compute_volume(fft_frame, N_, in_db=False, spl=False):
    volume = np.sum(np.abs(fft_frame) ** 2) / N_
    if in_db:  # decibels
        volume = 10 * np.log10(volume + 1e-10)
    if spl:  # sound pressure level (SPL)
        pref = 2e-5
        volume = 10 * np.log10(volume / pref + 1e-10)
    return volume


def visualise_volume(fft_frames, n_, N_, in_db=False, spl=True, fig=None, subplot_row=1, subplot_col=1):
    volumes = np.apply_along_axis(compute_volume, 1, fft_frames, N_=N_, in_db=in_db, spl=spl)
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(n_),
            y=volumes,
            mode='lines',
            name='Volume'
        ))
        fig.update_layout(
            title='Volume of Each Frame',
            xaxis_title='Frame Index',
            yaxis_title='Volume (dB)',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(n_), y=volumes, mode='lines', name='Volume'),
            row=subplot_row, col=subplot_col
        )


def visualise_all(frames, frame_rate, n_, N_, window_type=None, in_db=False, spl=True):
    fft_frames = transform_frames_to_frequency_domain(frames, frame_rate, N_, window_type=window_type)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05,
                        subplot_titles=("Frames in Frequency Domain", "Volume"))
    fig.update_layout(height=1800, showlegend=True)

    visualise_frames(fft_frames, frame_rate, n_, N_, fig=fig, subplot_row=1, subplot_col=1)
    visualise_volume(fft_frames, n_, N_, in_db=in_db, spl=spl, fig=fig, subplot_row=2, subplot_col=1)

    fig.layout.xaxis.update(title="Frequency (Hz)")
    fig.layout.yaxis.update(title="Magnitude")
    fig.layout.xaxis2.update(title="Frame index")
    if in_db or spl:
        fig.layout.yaxis2.update(title="Volume (dB)")
    fig.show()


def find_base_frequency(signal, sampling_frequency, min_freq=50, max_freq=400):
    real_cepstrum = np.real(np.fft.ifft(np.log(np.abs(np.fft.fft(signal)))))
    min_quefrency = int(sampling_frequency / max_freq)
    max_quefrency = int(sampling_frequency / min_freq)
    local_max_quefrency = np.argmax(real_cepstrum[min_quefrency:max_quefrency]) + min_quefrency
    base_frequency = sampling_frequency / local_max_quefrency

    return real_cepstrum, base_frequency


def visualise_ceps(audio, frame_rate, max_freq, min_freq, fig=None, subplot_row=1, subplot_col=1):
    ceps, base_freq = find_base_frequency(audio, frame_rate)
    if fig is None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=np.arange(min_freq, max_freq - min_freq),
            y=ceps,
            mode='lines',
            name='Cepstrum'
        ))
        fig.update_layout(
            title='Cepstrum of Each Sample',
            xaxis_title='Sample Index',
            yaxis_title='Cepstrum',
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(min_freq, max_freq - min_freq), y=ceps, mode='lines', name='Cepstrum'),
            row=subplot_row, col=subplot_col
        )
    print(base_freq)


def spectrogram(frames: np.ndarray, frame_rate: int, n_: int, N_: int) -> None:
    fft_frames: np.ndarray = transform_frames_to_frequency_domain(frames, frame_rate, N_)

    # Compute power spectral density
    power_spectral_density = np.abs(fft_frames) ** 2

    # Convert to decibels
    power_spectral_density_db = np.apply_along_axis(compute_volume, 1, fft_frames, N_=N_, in_db=False, spl=True)

    # Calculate the time and frequency vectors
    time_vector = np.arange(frames.shape[0]) * n_ / frame_rate
    freq_vector = np.arange(N_ // 2) * frame_rate / N_

    # Truncate power_spectral_density_db to half its size (due to symmetry)
    power_spectral_density_db = power_spectral_density_db[:, :N_ // 2]

    # Plot the spectrogram using seaborn
    plt.figure(figsize=(12, 6))
    sns.heatmap(power_spectral_density_db.T, cmap='viridis', xticklabels=False, yticklabels=False)
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency')
    plt.title('Spectrogram')
    plt.show()

