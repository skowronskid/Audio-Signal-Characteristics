from sys import displayhook
import wave
import numpy as np
from IPython.display import Audio
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


# Read and split the audio file into frames


def read_wave(path, display=True):
    with wave.open(path, 'rb') as wav_file:
        frame_rate = wav_file.getframerate()
        n_samples = wav_file.getnframes()
        samples = wav_file.readframes(n_samples)
        audio = np.frombuffer(samples, dtype=np.int16).astype(np.float32)/32768
    
    audio_time  = n_samples/frame_rate #in seconds
    if display:
        displayhook(Audio(data=audio, rate=frame_rate))
    return audio, frame_rate, audio_time, n_samples


def split_to_frames(audio, frame_rate, percent_frame_size=0.1, percent_hop_length=0.5):
    # default frame_size is 10% of the audio and default frame overlap is 50% overlap

    # naming convention: n_ - number of frames, N_ - number of samples in a frame
    # convention is consistent with "Cechy sygnalu audio w dziedzinie czasu.pdf"
    frame_size = int(percent_frame_size * frame_rate)
    hop_length = int(percent_hop_length * percent_frame_size * frame_rate)
    frames = []
    for i in range(0, len(audio), hop_length):
        frame = audio[i:i + frame_size]
        if len(frame) == frame_size:
            frames.append(frame)
    try:
        frames = np.stack(frames)
        n_ = frames.shape[0]
        N_ = frames.shape[1]
        return frames, n_, N_
    except:
        print("Error: frames are not the same size")


# Plot the audio
def plot_audio(audio, audio_time, fig=None, subplot_row=1, subplot_col=1):
    times = np.linspace(0, audio_time, num=audio.shape[0])

    if fig is None:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(x=times, y=audio, mode='lines'),
        )

        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude"
        )
        fig.show()
    else:  # if not None, then it is a subplot
        fig.add_trace(
            go.Scatter(x=times, y=audio, mode='lines'),
            row=subplot_row, col=subplot_col
        )


# cechy sygnału w dziedzinie czasu na poziomie ramki

def get_volume(audio, N_):
    return np.sqrt(np.sum(np.power(audio, 2)) / N_)


def plot_volumes(frames, n_, N_, fig=None, subplot_row=1, subplot_col=1):
    volumes = np.apply_along_axis(get_volume, 1, frames, N_=N_)
    if fig is None:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=volumes, mode='lines'),
        )

        fig.update_layout(
            title="Volume of audio frames",
            xaxis_title="Frame index",
            yaxis_title="Volume"
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=volumes, mode='lines'),
            row=subplot_row, col=subplot_col
        )


def get_ste(audio, N_):
    # ste - short time energy
    return get_volume(audio, N_) ** 2


def plot_ste(frames, n_, N_, fig=None, subplot_row=1, subplot_col=1):
    ste = np.apply_along_axis(get_ste, 1, frames, N_=N_)
    if fig is None:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=ste, mode='lines'),
        )

        fig.update_layout(
            title="Short Time Energy of audio frames",
            xaxis_title="Frame index",
            yaxis_title="Short Time Energy"
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=ste, mode='lines'),
            row=subplot_row, col=subplot_col
        )


def get_zcr(audio, N_):
    # ZCR - zero crossing rate
    return np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * N_)


def plot_zcr(frames, n_, N_, fig=None, subplot_row=1, subplot_col=1):
    zcr = np.apply_along_axis(get_zcr, 1, frames, N_=N_)
    if fig is None:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=zcr, mode='lines'),
        )

        fig.update_layout(
            title="Zero Crossing Rate of audio frames",
            xaxis_title="Frame index",
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=zcr, mode='lines'),
            row=subplot_row, col=subplot_col
        )


def get_sr(audio, N_, zcr_bound, volume_bound):
    # sr - silent ratio
    zrc = get_zcr(audio, N_)
    volume = get_volume(audio, N_)
    if zrc > zcr_bound and volume > volume_bound:
        return 1
    else:
        return 0


def plot_sr(frames, n_, N_, bounds=[0.1, 0.1], fig=None, subplot_row=1, subplot_col=1):
    sr = np.apply_along_axis(get_sr, 1, frames, N_=N_, zcr_bound=bounds[0], volume_bound=bounds[1])
    if fig is None:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=sr, mode='lines'),
        )

        fig.update_layout(
            title="Silent Ratio of audio frames",
            xaxis_title="Frame index",
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=sr, mode='lines'),
            row=subplot_row, col=subplot_col
        )


def get_vu(frames):
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    rms_db = 20 * np.log10(rms)
    return rms_db


def plot_vu(frames, n_, fig=None, subplot_row=1, subplot_col=1):
    vu = get_vu(frames)
    if fig is None:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=vu, mode='lines'),
        )

        fig.update_layout(
            title="Volume undulation",
            xaxis_title="Frame index",
            yaxis_title="RMS [db]",
        )
        fig.show()
    else:
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=vu, mode='lines'),
            row=subplot_row, col=subplot_col
        )


def get_f0(audio, l_, amdf=False):
    # F0 - fundamental frequency, częstotliwość tonu podstawowego
    # autocorrelation function by default
    # amdf - average magnitude difference function 
    if l_ > len(audio):
        raise ValueError("l_ must be smaller than the length of audio")

    if amdf:
        return np.sum(np.abs(audio[:-l_] - audio[l_:]))
    else:
        return np.sum(audio[:-l_] * audio[l_:])


def plot_f0(frames, l_, n_, amdf=False, fig=None, subplot_row=1, subplot_col=1):
    f0 = np.apply_along_axis(get_f0, 1, frames, l_=l_, amdf=amdf)

    if fig is None:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=f0, mode='lines'),
        )

        fig.update_layout(
            title="Fundamental frequency of audio frames",
            xaxis_title="Frame index",
        )
        fig.show()

    else:
        fig.add_trace(
            go.Scatter(x=np.arange(0, n_), y=f0, mode='lines'),
            row=subplot_row, col=subplot_col
        )


def plot_all(audio, audio_time, frames, l_, n_, N_, sr_bounds):
    fig = make_subplots(rows=7, cols=1, vertical_spacing=0.05,
                        subplot_titles=(
                        "Audio Waveform", "Volume", "Short Time Energy", "Zero Crossing Rate", "Silent Ratio",
                        "Fundamental Frequency", "Volume Undulation"))

    fig.update_layout(height=1800, showlegend=False)

    plot_audio(audio, audio_time, fig=fig, subplot_row=1, subplot_col=1)
    plot_volumes(frames, n_, N_, fig=fig, subplot_row=2, subplot_col=1)
    plot_ste(frames, n_, N_, fig=fig, subplot_row=3, subplot_col=1)
    plot_zcr(frames, n_, N_, fig=fig, subplot_row=4, subplot_col=1)
    plot_sr(frames, n_, N_, fig=fig, subplot_row=5, subplot_col=1, bounds=sr_bounds)
    plot_f0(frames, l_, n_, fig=fig, subplot_row=6, subplot_col=1, amdf=False)
    plot_vu(frames, n_, fig=fig, subplot_row=7, subplot_col=1)
    fig.layout.xaxis.update(title="Time (s)")
    fig.layout.xaxis2.update(title="Frame index")
    fig.layout.xaxis3.update(title="FraFrame indexmes")
    # fig.layout.yaxis2.update(title="Volume (dB)")
    fig.layout.xaxis4.update(title="Frame index")
    # fig.layout.yaxis2.update(title="Volume (dB)")
    fig.layout.xaxis5.update(title="Frame index")
    # fig.layout.yaxis2.update(title="Volume (dB)")
    fig.layout.xaxis6.update(title="Frame index")
    fig.layout.xaxis7.update(title="Frame index")

    fig.show()


# cechy sygnału audio w dziedzinie czasu na poziomie klipu

def get_avg_amplitue(audio):
    return np.mean(np.abs(audio))


def get_vstd(audio):
    # vstd - volume standard deviation normalized by the maximum value
    return np.std(audio) / np.max(np.abs(audio))


def get_vdr(audio):
    # vdr - volume dynamic range 
    return (np.max(audio) - np.min(audio)) / np.max(audio)


def get_energy_entropy(frames):
    energy = np.sum(np.square(frames), axis=1)
    energy_dist = energy / np.sum(energy)
    return -np.sum(energy_dist * np.log2(energy_dist))


def get_zstd(frames, N_):
    zcr_values = np.apply_along_axis(get_zcr, axis=1, arr=frames, N_=N_)
    zcr_std = np.std(zcr_values)
    return zcr_std


def print_info(audio, audio_time, frame_rate, frames, n_, N_):
    print(f"Audio length: {np.format_float_positional(audio_time, 2)} s")
    print(f"Frame rate: {frame_rate} Hz")
    print(f"Number of frames: {n_}")
    print(f"Number of samples in a frame: {N_}")
    print(f"Frame length: {np.format_float_positional(N_ / frame_rate, 3)}s")
    print(f"Avarege amplitude: {np.format_float_positional(get_avg_amplitue(audio), precision=1)}")
    print(f"VSTD: {np.format_float_positional(get_vstd(audio), precision=4)}")
    print(f"VDR: {np.format_float_positional(get_vdr(audio), precision=4)}")
    print(f"Energy entropy: {np.format_float_positional(get_energy_entropy(frames), precision=4)}")
    print(f"ZSTD: {np.format_float_positional(get_zstd(frames, N_), precision=4)}")


# bazujace na energii


def split_to_sec_frames(audio, frame_rate):
    return np.split(audio, np.arange(frame_rate, len(audio), frame_rate))


def get_lstr(frame_sec, frame_rate, percent_frame_size, percent_hop_length):
    frames, n_, N_ = split_to_frames(frame_sec, frame_rate, percent_frame_size, percent_hop_length)
    stes = np.apply_along_axis(get_ste, 1, frames, N_=N_)
    ste_mean = np.mean(stes)
    return np.sum((0.5 * ste_mean > stes) + 1) / (2 * len(frame_sec))


def plot_lstr(audio, frame_rate, percent_frame_size, percent_hop_length):
    frames_sec = split_to_sec_frames(audio, frame_rate)
    lstr = []
    for frame_sec in frames_sec[:-1]:
        lstr.append(get_lstr(frame_sec, frame_rate, percent_frame_size, percent_hop_length))
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(x=np.arange(0, len(frames_sec)), y=lstr, mode='lines'),
    )

    fig.update_layout(
        title="Low short time energy ratio of audio frames",
        yaxis_title="Ratio",
        xaxis_title="Second",
    )
    fig.show()


def get_hzcrr(frame_sec, frame_rate, percent_frame_size, percent_hop_length):
    frames, n_, N_ = split_to_frames(frame_sec, frame_rate, percent_frame_size, percent_hop_length)
    zcrs = np.apply_along_axis(get_zcr, 1, frames, N_=N_)
    zcr_mean = np.mean(zcrs)
    return np.sum((1.5 * zcr_mean < zcrs) + 1) / (2 * len(frame_sec))


def plot_hzcrr(audio, frame_rate, percent_frame_size, percent_hop_length):
    frames_sec = split_to_sec_frames(audio, frame_rate)
    hzcrr = []
    for frame_sec in frames_sec[:-1]:
        hzcrr.append(get_hzcrr(frame_sec, frame_rate, percent_frame_size, percent_hop_length))
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(x=np.arange(0, len(frames_sec)), y=hzcrr, mode='lines'),
    )

    fig.update_layout(
        title="High zero crossing rate ratio of audio frames",
        yaxis_title="Ratio",
        xaxis_title="Second",
    )
    fig.show()


def plot_spectrum(audio, audio_time, frame_rate):
    plt.figure(figsize=(10, 6))
    plt.specgram(audio, Fs=frame_rate)
    plt.title('Audio spectrogram')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.xlim(0, audio_time)
    plt.colorbar()
    plt.show()
