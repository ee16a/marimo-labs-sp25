import csv
import numpy as np
#import sounddevice
#import pyaudio
from tqdm.notebook import trange
from IPython import display
from scipy.signal import butter, filtfilt
import marimo as mo

def read_csv(filename):
    return np.genfromtxt(filename, dtype=np.float64, delimiter=",")


def train_test_split(data, split_ratio=0.7):
    """Splits data into training and test set according to the split_ratio.
    Arguments:
        data: dataset as a numpy array
        split_ratio: fraction of dataset to split as training data (must be between 0 and 1)
    Returns:
        Training Data (size = split_ratio * size of original dataset)
        Test Data (size = (1 - split_ratio) * size of original dataset)
    """
    np.random.shuffle(data)
    train_data, test_data = data[:int(split_ratio *
                                      len(data)), :], data[int(split_ratio *
                                                               len(data)):, :]

    return train_data, test_data

def format_constant_c(name, constant):
    # <Insert smug remark about left-pad>
    if len(name) < 37:
        padding = " " * (38 - len(name) - len("#define "))
    else:
        padding = "\t"
    return "#define {}{}{}".format(name, padding, constant)


def format_array_c(name, array, dtype="float"):
    contents = ", ".join(map(str, array))
    return "{} {}[{}] = {{{}}};".format(dtype, name, len(array), contents)

# util functions for recording audio
def record_audio(seconds=3, rate=44100, chunk=1024):
    """Record audio for a given number of seconds."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    frames = []

    for _ in mo.status.progress_bar(
        range(int(rate / chunk * seconds)),
        title="Recording",
        show_eta=True,
        show_rate=True,
        remove_on_exit=True
    ):
        data = stream.read(chunk)
        frames.append(np.frombuffer(data, dtype=np.int16))

    print("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.hstack(frames)

def create_recording_csv(rate=5400, chunk=1024, record_seconds=3, num_recordings=50):
    filename = input("Enter the CSV file name (it will be created): ")
    if not filename.endswith('.csv'):
        print("Warning: filename does not end with .csv. Appending .csv to the filename.")
        filename = filename + '.csv'

    recording_count = 0

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        try:
            while recording_count < num_recordings:
                user_input = input("Press Enter to start recording, or type 'd' and then Enter to remove the last entry, or type 'stop' and then Enter to stop recording: ")

                if user_input == '':
                    # Record audio
                    audio_data = record_audio(seconds=record_seconds, rate=rate, chunk=chunk)
                    writer.writerow(audio_data)
                    file.flush()

                    recording_count += 1

                    display.clear_output()
                    print(f"Audio data saved to CSV. Recorded {recording_count}/{num_recordings}. Length of vector: {len(audio_data)}\n")

                elif user_input.lower() == 'd': 
                    if recording_count > 0:
                        # Remove the last entry
                        with open(filename, mode='r') as f:
                            lines = f.readlines()
                        with open(filename, mode='w') as f:
                            f.writelines(lines[:-1])

                        recording_count -= 1
                        display.clear_output()
                        print(f"Last recording removed. Recorded {recording_count}/{num_recordings}.")
                    else:
                        print("No entries to delete.")
                elif user_input.lower() == 'stop': 
                    break

        except KeyboardInterrupt:
            print("Data collection stopped.")

    print("Data collection stopped. Saved to " + filename)
    
def envelope(signal, sampling_rate, cutoff_frequency=1000):
    """
    Applies an envelope detector to an audio signal.

    Parameters:
        signal (numpy array): The input audio signal.
        sampling_rate (int): Sampling rate of the audio signal in Hz.
        cutoff_frequency (float): Cutoff frequency for the low-pass filter in Hz (default is 10.0).

    Returns:
        numpy array: The envelope of the input signal.
    """
    # Rectify the signal (take absolute value)
    rectified_signal = np.abs(signal)
    
    # Design a low-pass Butterworth filter
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_frequency / nyquist
    b, a = butter(2, normal_cutoff, btype='low', analog=False)

    # Apply the low-pass filter to the rectified signal
    envelope_signal = filtfilt(b, a, rectified_signal)
    
    return envelope_signal

# mel scaled STFT helper functions

# Create Mel Filter Bank
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)

def mel_filter_bank(sample_rate, n_fft, n_mels):
    mel_min = 0
    mel_max = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Convert Hz points to FFT bin numbers
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    # Create the filter bank matrix
    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for i in range(1, len(bin_points) - 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        for j in range(left, center):
            filters[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            filters[i - 1, j] = (right - j) / (right - center)

    return filters

def apply_mel_filter_helper(Zxx, mel_filter):
    power_spectrogram = np.abs(Zxx)**2
    mel_spectrogram = np.dot(mel_filter, power_spectrogram)
    log_mel_spectrogram = power_to_db(mel_spectrogram)
    return log_mel_spectrogram

def apply_mel_filter(Zxxs, mel_filter):
    mel_spec_lst = []
    for val in Zxxs:
        log_mel_spectrogram = apply_mel_filter_helper(val, mel_filter)
        mel_spec_lst.append(log_mel_spectrogram)
    return mel_spec_lst

def power_to_db(S, ref=1.0, amin=1e-10):
    S_db = 10.0 * np.log10(np.maximum(amin, S / ref))
    return S_db

def mel_frequencies(n_mels, sample_rate, n_fft):
    mel_min = 0
    mel_max = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)  # Add 2 for the edges
    hz_points = mel_to_hz(mel_points)  # Convert mel points to hz
    return hz_points[1:-1]  # We ignore the first and last points (0 Hz and Nyquist)
