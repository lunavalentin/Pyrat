import numpy as np # for matrix operations and fast math functions

import glob
import os as os # for operating system path manipulation

import scipy.signal as signal # for signal processing functions
from scipy.signal import butter, filtfilt, fftconvolve, get_window, resample
from scipy.signal.windows import hann
from scipy.special import erfc
from scipy import stats

import matplotlib.pyplot as plt # for plotting things
import IPython.display as ipd # displaying audio in the python notebook
from IPython.display import clear_output
from matplotlib.gridspec import GridSpec

import soundfile as sf # for loading and writing audio files
from pydub import AudioSegment
from pydub.utils import mediainfo
import subprocess
from scipy.io.wavfile import write
from scipy.io import wavfile

def list_wav_files_in_folder(folder_path):
    """
    Returns a list of .wav file paths from the specified folder.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing .wav files.

    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

    # Construct the search pattern
    search_pattern = os.path.join(folder_path, '*.wav')
    
    # Gather all matching .wav files
    wav_files = sorted(glob.glob(search_pattern))

    return wav_files

def get_scalefactor_from_wav(wav_file_path):
    """
    Extract the 'scalefactor' value from a WAV file's metadata comment tag.
    
    This function relies on 'pydub.utils.mediainfo' to retrieve metadata
    from the WAV file. If the metadata contains a 'TAG' field with a 
    'comment' key, and the string 'scalefactor=' is found, the substring 
    that follows will be returned as a float. If no scalefactor is found,
    the function returns None.

    :param wav_file_path: The path to the WAV file.
    :type wav_file_path: str
    :return: The scalefactor as a float, or None if the scalefactor is not found.
    :rtype: float or None
    """
    info = mediainfo(wav_file_path)
    
    # Check if 'TAG' exists in the media info and if the comment exists
    if 'TAG' in info and 'comment' in info['TAG']:
        comment = info['TAG']['comment']
        
        # Extract the scalefactor from the comment string
        if 'scalefactor=' in comment:
            scalefactor = comment.split('scalefactor=')[1]
            return float(scalefactor)  # Return the scalefactor as a float

    # Return None if the scalefactor is not found
    return None

def add_scalefactor_to_wav(wav_file_path, scalefactor, output_file_path):
    """
    Adds a 'scalefactor=...' comment metadata tag to a WAV file using FFmpeg, and
    saves the result to a new file without re-encoding the audio data.

    Parameters
    ----------
    wav_file_path : str
        The path to the input WAV file.
    scalefactor   : float
        The scalefactor value to embed in the WAV file's metadata comment field.
    output_file_path : str
        The output path for the WAV file with the added comment.

    Returns
    -------
    None
        Prints the FFmpeg command for debugging and indicates success or error
        messages from FFmpeg.

    """
    # Construct the ffmpeg command to add the comment metadata
    ffmpeg_command = [
        'ffmpeg', '-y', '-i', wav_file_path,
        '-metadata', f'comment=scalefactor={scalefactor}',
        '-codec', 'copy',  # Ensures the audio data is not re-encoded
        output_file_path
    ]

    # Print the command for debugging
    print("Running ffmpeg command: ", ' '.join(ffmpeg_command))

    # First run (can be removed if you only want to run once)
    subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Second run, storing the result
    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FFmpeg error:", result.stderr.decode())
    else:
        print("FFmpeg command ran successfully.")

def async_4chan_sweep_to_ir(wav_file, sig_or, duration, output_folder, temp):
    """
    Processes a single multi-channel measurement WAV file by performing 
    deconvolution against a known original chirp (`sig_or`) to obtain 
    impulse responses (IRs).

    Specially made to process H3VR chirps into IRs

    Steps:
    1. Loads the WAV data and pads it if needed to match the reference signal length.
    2. Performs an FFT-based deconvolution to compute the impulse response for each channel.
    3. Identifies the direct sound arrival, trimming the IR to start just before that point.
    4. Cuts the IR to a specified `duration` in seconds.
    5. Normalizes and writes the IR to a temporary WAV file with 32-bit floating-point precision.
    6. Adds a 'scalefactor' metadata tag to the final file using FFmpeg.
    7. Returns the path of the final WAV file and prints processing info.

    Parameters
    ----------
    wav_file : str
        Path to the input multi-channel WAV file.
    sig_or : numpy.ndarray
        The reference chirp (or original signal) used for deconvolution.
    duration : float
        Length of the final IR (in seconds) to keep after trimming.
    output_folder : str
        Destination folder for the final IR WAV file.
    temp : str
        Temporary folder to store intermediate files before metadata injection.

    Returns
    -------
    final : str
        The path to the processed IR WAV file with the scalefactor metadata added.

    Notes
    -----
    - This function expects the input file to have the same sample rate 
      as `sig_or` or at least a matching sample rate for the measurement.
    - Ensure FFmpeg is installed and available in the system's PATH for the
      metadata tagging step.
    """
    # Check file existence
    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"File {wav_file} not found.")

    print(f'Now processing: {wav_file}')
    # Read the measurement signal
    sig, fs = sf.read(wav_file)

    # Calculate how many samples to pad
    s = int(abs(len(sig_or) - len(sig)))

    # If input is mono, reshape to (samples, 1) to handle consistently
    if len(sig.shape) == 1:
        sig = sig[:, np.newaxis]

    # Pad the measurement signal to match the reference chirp length
    zeros_to_pad = np.zeros((s, sig.shape[1]))
    padded_signal = np.concatenate((zeros_to_pad, sig), axis=0)

    # Pad the reference signal (sig_or)
    zeros_to_pad = np.zeros(s)
    padded_signal_or = np.concatenate((zeros_to_pad, sig_or), axis=0)

    # Transpose for channel-wise processing
    padded_signal = padded_signal.transpose()

    ir_channels = []

    # Deconvolution per channel
    for i in range(len(padded_signal)):
        print(f'Preparing channel {i + 1} for deconvolution...')

        nbins = 2 ** (int(np.ceil(np.log2(len(sig_or) + len(padded_signal[i])))))
        fft_response = np.fft.fft(padded_signal[i], 2 * nbins)
        fft_chirp = np.fft.fft(padded_signal_or, 2 * nbins)

        # Avoid division by zero by adding a small epsilon
        ir_chan = np.real(np.fft.ifft(fft_response / (fft_chirp + 1e-10)))

        # Identify direct sound arrival on the first channel (modify as needed)
        if i == 0:
            scale_factor_est = np.max(np.abs(ir_chan)) * 32768 / 32767
            # Find first sample above 0.6 (normalized), then shift by fs/3 samples
            direct_sound_arrival_sample = np.abs(
                np.where((np.abs(ir_chan) / scale_factor_est) > 0.6)[0][0] - int(fs / 3)
            )

        # Trim IR to start just after direct sound arrival
        ir_chan = ir_chan[direct_sound_arrival_sample:]

        # Keep only the first `ntaps` samples (based on desired duration)
        ntaps = round(duration * fs)
        ir_chan = ir_chan[:ntaps]

        ir_channels.append(ir_chan)
        print(f'Channel {i + 1} done.')

    # Stack channels side by side: (samples, channels)
    ir_combined = np.column_stack(ir_channels)

    # Normalize the combined IR
    scalefactor = np.max(np.abs(ir_combined)) * 32768 / 32767
    print('Max. abs(IR) =', np.max(np.abs(ir_combined)))
    ir_normalized = ir_combined / scalefactor

    # Prepare output paths
    base_file_name, _ = os.path.splitext(os.path.basename(wav_file))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    scalefactor_str = f'{scalefactor:.8f}'
    output_file_path = os.path.join(temp, f'IR_nsc_{base_file_name}.wav')
    final = os.path.join(output_folder, f'IR_n_dhs_{base_file_name}.wav')

    # Write IR to a temporary WAV file
    sf.write(output_file_path, ir_normalized, fs, format='wav', subtype='PCM_32')

    # Add scale factor metadata (make sure add_scalefactor_to_wav is imported)
    add_scalefactor_to_wav(output_file_path, scalefactor_str, final)

    # Verify that scalefactor is embedded
    embedded_sf = get_scalefactor_from_wav(final)
    print(f"Scalefactor from '{final}': {embedded_sf}")

    print(f"{wav_file} => {final} (Processed)")

    return final

def split_and_save_channels(input_wav, output_folder):
    """
    Splits a 4-channel (or multi-channel) WAV file into separate 
    mono WAV files for the first 4 channels and saves them to disk.

    Parameters
    ----------
    input_wav : str
        Path to the multi-channel WAV file.
    output_folder : str
        Path to the folder where the individual channel files will be saved.

    Raises
    ------
    ValueError
        If the input file is not multi-channel or has fewer than 4 channels.
    FileNotFoundError
        If the input WAV file is not found.

    Notes
    -----
    - The output files are saved in 32-bit floating-point format (`PCM_32`).
    - If the file has more than 4 channels, this function will still 
      only process channels 1 through 4.
    - If `output_folder` does not exist, it will be created.
    """
    # Check if the file exists
    if not os.path.isfile(input_wav):
        raise FileNotFoundError(f"Input file '{input_wav}' not found.")

    # Read the multi-channel WAV file
    data, fs = sf.read(input_wav)

    # Check if the file has at least 4 channels
    if data.ndim == 1:
        raise ValueError("Input file is not multi-channel (only 1 channel).")
    elif data.shape[1] < 4:
        raise ValueError(f"Input file only has {data.shape[1]} channel(s). 4 channels are required.")

    # Ensure the output folder exists or create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Extract and save each of the first 4 channels
    base_file_name, _ = os.path.splitext(os.path.basename(input_wav))
    for i in range(4):
        channel_data = data[:, i]
        output_file_name = f"ch_{i+1}_{base_file_name}.wav"
        output_file_path = os.path.join(output_folder, output_file_name)

        # Write the mono WAV file for this channel
        sf.write(output_file_path, channel_data, fs, format='wav', subtype='PCM_32')
        print(f"Saved: {output_file_path}")

def normd_sweep_to_ir(wav_file, orig_sweep, duration, output_folder, temp):
    """
    Process a single WAV file to generate an impulse response (IR), 
    normalized according to a reference sweep signal.

    This function:
        1. Reads the reference sweep from `orig_sweep`.
        2. Reads the measurement WAV file (mono or stereo). If stereo, 
           only the left channel is used.
        3. Zero-pads the measurement signal and reference sweep to match
           a power-of-two FFT size.
        4. Performs FFT-based deconvolution to obtain the IR.
        5. Trims the IR to `duration` seconds.
        6. Normalizes the IR so its peak amplitude is +/-1 in float terms,
           corresponding to a `scalefactor` for integer representation.
        7. Writes the normalized IR to `output_folder` as a 32-bit float WAV file.

    Parameters
    ----------
    wav_file : str
        Path to the input measurement WAV file.
    orig_sweep : str
        Path to the reference sweep WAV file for deconvolution.
    duration : float
        Length (in seconds) of the final IR to keep after trimming.
    output_folder : str
        Destination folder where the processed IR will be saved.

    Returns
    -------
    None
        Outputs a single IR WAV file to `output_folder` and prints
        progress messages.
    """
    # --- 1) Read the reference sweep ---
    if not os.path.exists(orig_sweep):
        raise FileNotFoundError(f"Reference sweep file '{orig_sweep}' not found.")
    sig_or, fs_sweep = sf.read(orig_sweep)
    print(f"Reference sweep loaded: {orig_sweep}, length={len(sig_or)}, fs={fs_sweep}")

    # --- 2) Read the measurement WAV file ---
    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"Measurement file '{wav_file}' not found.")
    sig, fs = sf.read(wav_file)
    if fs != fs_sweep:
        print(f"Warning: '{wav_file}' has sample rate {fs}, "
              f"but reference sweep is {fs_sweep} Hz. "
              "They should match for accurate deconvolution.")

    # If stereo, use only the left channel
    if sig.ndim > 1 and sig.shape[1] == 2:
        sig = sig[:, 0]  # left channel

    print(f"\nNow processing: {wav_file}")

    # --- 3) Determine FFT size and pad signals ---
    # Add lengths, find nearest power of 2, then subtract 1 from exponent (optional).
    nbins = 2 ** (int(np.ceil(np.log2(len(sig_or) + len(sig)))) - 1)
    
    padded_chirp = np.pad(sig_or, (0, nbins - len(sig_or)))
    padded_response = np.pad(sig, (0, nbins - len(sig)))

    # --- 4) Perform FFT-based deconvolution ---
    fft_response = np.fft.fft(padded_response, 2 * nbins)
    fft_chirp = np.fft.fft(padded_chirp, 2 * nbins)
    # Replace zeros in fft_chirp to avoid /0
    fft_chirp_safe = np.where(fft_chirp == 0, 1e-10, fft_chirp)

    ir_full = np.real(np.fft.ifft(fft_response / fft_chirp_safe))

    # --- 5) Trim the IR ---
    ntaps = round(duration * fs)
    ir_cropped = ir_full[:ntaps]

    # --- 6) Normalize the IR ---
    peak_val = np.max(np.abs(ir_cropped))
    if peak_val == 0:
        scalefactor = 1.0
    else:
        scalefactor = peak_val * 32768 / 32767

    print("Max. abs(IR) =", peak_val)
    print("Computed scalefactor =", scalefactor)

    ir_normalized = ir_cropped / scalefactor
    # Prepare output paths
    base_file_name, _ = os.path.splitext(os.path.basename(wav_file))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- Ensure output folder exists ---
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # --- Write the IR to a WAV file ---
    scalefactor_str = f'{scalefactor:.8f}'
    output_file_path = os.path.join(temp, f'IR_nsc_{base_file_name}.wav')
    final = os.path.join(output_folder, f'IR_n_dhs_{base_file_name}.wav')

    # Write IR to a temporary WAV file
    sf.write(output_file_path, ir_normalized, fs, format='wav', subtype='PCM_32')

    # Add scale factor metadata (make sure add_scalefactor_to_wav is imported)
    add_scalefactor_to_wav(output_file_path, scalefactor_str, final)

    # Verify that scalefactor is embedded
    embedded_sf = get_scalefactor_from_wav(final)
    print(f"Scalefactor from '{final}': {embedded_sf}")

    print(f"{wav_file} => {final} (Processed)")

###########################################################################################################################

def disp_ir(ir, fs, nfft=2048, overlap=4):
    """
    Display a time-domain waveform and a log-scaled spectrogram of the given IR (or any signal).
    
    Parameters
    ----------
    ir : numpy.ndarray
        The signal or IR array. Can be 1D (mono) or 2D (multi-channel).
    fs : float
        Sampling rate in Hz.
    nfft : int, optional
        FFT size for the spectrogram (default: 2048).
    overlap : int, optional
        Number of overlapping samples between segments in the spectrogram (default: 4).
    
    Returns
    -------
    None
        Displays a matplotlib figure with two subplots:
        1) Time-domain waveform
        2) Log-frequency spectrogram (dB scale)
    """
    # Ensure ir is a NumPy array
    ir = np.asarray(ir)
    
    # If IR has multiple channels, we'll plot the first channel in the spectrogram
    if ir.ndim > 1:
        # multi-channel: shape = (samples, channels)
        # We'll transpose so that indexing becomes [channel, samples].
        sig_spectr = ir.T[0]  # first channel
    else:
        # single-channel
        sig_spectr = ir
    
    # Create a time vector for the entire signal
    time_vec = np.arange(len(ir)) / fs
    
    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8),
        gridspec_kw={'height_ratios': [1, 2]}
    )
    
    # 1) Plot the time-domain waveform
    ax1.plot(time_vec, ir if ir.ndim == 1 else ir, '-')
    ax1.grid()
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim([0, len(ir) / fs])
    
    # Adjust vertical range if the signal is normalized or within [-1, 1]
    # Otherwise, the default autoscale might be better.
    # If you regularly have signals in [-1, 1], you can uncomment:
    # ax1.set_ylim([-1, 1])
    
    # 2) Create the spectrogram on the first channel
    Pxx, freqs, t_spec, im = ax2.specgram(
        sig_spectr,
        NFFT=nfft,
        Fs=fs,
        window=np.hanning(nfft),
        noverlap=overlap,
        pad_to=nfft * 2
    )
    
    # Convert to dB scale, clamp values below -120 dB
    epsilon = 1e-10
    Pxx_dB = 20 * np.log10((Pxx + epsilon) / np.max(Pxx + epsilon))
    Pxx_dB = np.where(Pxx_dB < -120, -120, Pxx_dB)
    
    # Plot the spectrogram in dB
    cores = ax2.pcolormesh(t_spec, freqs, Pxx_dB, vmin=-120, vmax=0, shading='auto')
    
    # Configure the axes and colorbar
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    cbar = plt.colorbar(cores, orientation="horizontal", pad=0.15)
    cbar.set_label('Power (dB)')
    cbar.mappable.set_clim(vmin=-80, vmax=0)
    
    ax2.set_yscale('log')
    ax2.set_ylim([20., 20000.])
    ax2.set_xlim([0, len(ir) / fs])
    
    plt.subplots_adjust(hspace=0.3)
    plt.show()

def identify_direct_sound_arrival_max(ir):
    """
    Identify the index of the direct sound arrival in an impulse response 
    by searching for the sample with the largest absolute amplitude.
    
    :param ir: A NumPy array representing the impulse response.
    :return: An integer representing the index of the direct sound 
             arrival sample (i.e., the maximum absolute value in 'ir').
    """
    direct_sound_arrival_sample = np.argmax(np.abs(ir))
    return direct_sound_arrival_sample

def identify_direct_sound_arrival_threshold(ir, threshold):
    """
    Identify the index of the direct sound arrival in an impulse response by checking
    when its normalized amplitude first exceeds a specified threshold.

    This function scales the impulse response based on the maximum absolute value,
    and then determines the first sample index where the normalized amplitude
    exceeds the given threshold.

    :param ir: A NumPy array representing the impulse response.
    :param threshold: A floating-point threshold (0 <= threshold <= 1) at which
                      the direct sound arrival is considered to occur.
    :return: An integer index of the first sample exceeding the threshold.
    """
    scale_factor = np.max(np.abs(ir)) * 32768 / 32767
    direct_sound_arrival_sample = np.where((np.abs(ir) / scale_factor) > threshold)[0][0]
    return direct_sound_arrival_sample

def snr_calc(ir, fs, threshold=0.7):
    """
    Compute the signal-to-noise ratio (SNR) in dB for a given impulse response (IR).
    
    The function locates the direct sound arrival using a threshold on the normalized IR,
    then treats a short segment (0.02 s) after the direct arrival as the "signal" portion,
    and the segment before the direct arrival as "noise."
    
    Parameters
    ----------
    ir : numpy.ndarray
        The impulse response data (1D or multi-channel). If multi-channel, 
        only the first channel is used.
    fs : float
        The sample rate of 'ir' in Hz.
    threshold : float, optional
        The normalized amplitude threshold for detecting the direct sound arrival 
        (default is 0.7).
    
    Returns
    -------
    float
        The computed SNR in decibels (dB).

    Raises
    ------
    ValueError
        If the IR is empty or the detected direct sound arrival index goes out of range.

    """
    
    def identify_direct_sound_arrival_threshold(ir_array, thresh):
        # Avoid zero maximum case
        peak_val = np.max(np.abs(ir_array))
        if peak_val == 0:
            raise ValueError("Impulse response is empty or all zeros.")
        scale_factor = peak_val * 32768 / 32767
        indices_above_thresh = np.where((np.abs(ir_array) / scale_factor) > thresh)[0]
        if not len(indices_above_thresh):
            raise ValueError("No sample exceeds the specified threshold.")
        return indices_above_thresh[0]

    # If IR has multiple channels, use only the first channel
    if ir.ndim > 1:
        ir = ir[:, 0]

    if ir.size == 0:
        raise ValueError("Impulse response array is empty.")

    # Identify direct sound arrival
    ir_start_sample = identify_direct_sound_arrival_threshold(ir, threshold)

    # Define the signal and noise segments
    ir_end_sample = ir_start_sample + int(0.02 * fs)  # 20 ms after direct arrival
    if ir_end_sample > len(ir):
        ir_end_sample = len(ir)  # clamp to end if needed

    noise_start_sample = 0
    noise_end_sample = ir_start_sample

    # Extract the signal and noise segments
    ir_seg = ir[ir_start_sample:ir_end_sample]
    noise_seg = ir[noise_start_sample:noise_end_sample]

    # Compute average power for each segment
    signal_power = np.sum(np.square(ir_seg)) / len(ir_seg) if len(ir_seg) else 0
    noise_power = np.sum(np.square(noise_seg)) / len(noise_seg) if len(noise_seg) else 1e-12

    # Calculate SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)
    print(f"SNR: {snr_db:.2f} dB")

    return snr_db

def F2smospa(fvec, F2in, octfrac, fstart, fend, npoints):
    """
    Applies octave-band related smoothing to a squared transfer-function (TF) 
    magnitude spectrum.

    Parameters
    ----------
    fvec     : array_like
        Input frequency vector (should be equally spaced or at least sorted).
    F2in     : array_like
        Squared TF magnitude (|F|^2).
    octfrac  : float
        Octave band fraction (e.g., 1 for one-octave bands, 3 for third-octave bands).
    fstart   : float
        Starting frequency of the output vector (Hz).
    fend     : float
        Ending frequency of the output vector (Hz).
    npoints  : int
        Number of frequency values in the output vector.

    Returns
    -------
    F2out    : numpy.ndarray
        Smoothed squared TF magnitude.
    fvecout  : numpy.ndarray
        Frequency values of the output vector (log-spaced).
    """
    # Check if frequencies are (roughly) equally spaced
    dfvec = np.diff(fvec)
    if np.std(dfvec) < 1e-6:
        linfreq = 1
        df = fvec[1] - fvec[0]
    else:
        linfreq = 0

    # Create the log-spaced frequency vector
    fvecout = np.logspace(np.log10(fstart), np.log10(fend), npoints)
    
    # Precompute smoothing factors
    nfreqsperdecade = npoints / (np.log10(fend) - np.log10(fstart))
    nfreqsperoctave = nfreqsperdecade * np.log10(2)  # for reference if needed
    freqmultfac = 2 ** (1 / (2 * octfrac))
    flosmo = fvecout / freqmultfac
    fhismo = fvecout * freqmultfac

    F2out = np.zeros(npoints)
    F2in = np.abs(F2in)

    if linfreq == 1:
        # For linearly spaced frequency vectors
        iv1 = np.round((flosmo - fvec[0]) / df).astype(int)
        iv2 = np.round((fhismo - fvec[0]) / df).astype(int)
        
        for ii in range(npoints):
            if iv1[ii] < 0 or iv2[ii] >= len(F2in):  # Boundary check
                F2out[ii] = np.nan
            else:
                F2out[ii] = np.mean(F2in[iv1[ii]:iv2[ii]])
    else:
        # For non-linear spacing (or if spacing is uncertain)
        for ii in range(npoints):
            iv = np.where((fvec >= flosmo[ii]) & (fvec <= fhismo[ii]))[0]
            if len(iv) == 0:
                F2out[ii] = np.nan  # No data in this frequency range
            else:
                F2out[ii] = np.mean(F2in[iv])

    return F2out, fvecout

def smoothed_frequency_response(ir, sample_rate, start_time, end_time, threshold,
                                octfrac=1, fstart=20, fend=20000, npoints=100):
    """
    Compute a smoothed frequency response (magnitude squared) from a segment 
    of the provided impulse response, using octave-band smoothing.

    Parameters
    ----------
    ir          : array_like
        The impulse response data (1D or multi-channel).
    sample_rate : float
        The sampling rate in Hz.
    start_time  : float
        Start time (in seconds) for the segment to be analyzed, measured
        from the detected direct sound arrival.
    end_time    : float
        End time (in seconds) for the segment to be analyzed, measured
        from the detected direct sound arrival.
    threshold   : float
        Threshold (0 to 1) used to detect direct sound arrival in the IR.
    octfrac     : float, optional
        Octave band fraction (default is 1, meaning full octaves).
    fstart      : float, optional
        Starting frequency for smoothing (Hz). Default is 20 Hz.
    fend        : float, optional
        Ending frequency for smoothing (Hz). Default is 20,000 Hz.
    npoints     : int, optional
        Number of frequency points in the output vector (default is 100).

    Returns
    -------
    F2out  : numpy.ndarray
        The smoothed squared magnitude response over the selected frequency range.
    fvecout: numpy.ndarray
        The frequency vector corresponding to the smoothed response.
    """
    # If IR has multiple channels, convert to mono
    if len(ir.shape) > 1:
        ir = np.mean(ir, axis=1)

    # Identify direct sound arrival and slice the IR
    istart = identify_direct_sound_arrival_threshold(ir, threshold)
    start_sample = int(start_time * sample_rate) + istart
    end_sample = int(end_time * sample_rate) + istart

    # Extract segment
    data_segment = ir[start_sample:end_sample]

    # Compute squared magnitude of the FFT (transfer function)
    nfft = 2 ** 18  # you can adjust if needed
    F = np.fft.fft(data_segment, nfft)
    # Single-sided frequency vector
    fvec = np.arange(nfft // 2 - 1) * (sample_rate / nfft)
    F2in = np.abs(F[:nfft // 2]) ** 2  # consider only positive frequencies

    # Apply octave-band smoothing
    F2out, fvecout = F2smospa(fvec, F2in, octfrac, fstart, fend, npoints)

    print("FR response done!")
    return F2out, fvecout

def decaytime(ir, fs, dec, f_min=20, f_max=22, fbi=None, zetai=1.0):
    """
    Compute reverberation time (RT60-like metric) in user-specified frequency bands 
    for a given impulse response.

    The process:
    1. Identify the direct sound arrival in the IR and trim the first 20 ms before it.
    2. Filter the IR into octave or user-defined bands.
    3. Compute a smoothed energy envelope for each band.
    4. Estimate noise floor and fit a linear decay to determine the time it takes 
       to decay 'dec' dB (e.g., 60 dB for RT60).

    Parameters
    ----------
    ir : array_like
        The impulse response data (1D or multi-channel). 
        If multi-channel, the mean of all channels is used.
    fs : float
        Sample rate in Hz of the impulse response.
    dec : float
        Decay level in dB for RT calculation (e.g., 60 for RT60).
    f_min : float, optional
        Minimum frequency (Hz) of bands to include in the analysis (default = 20 Hz).
    f_max : float, optional
        Maximum frequency (Hz) of bands to include in the analysis (default = 22 Hz).
        *Note*: in the original code, f_max=22 likely means 22 kHz if your 
        IR is at 44.1 kHz or 48 kHz sampling. Adjust as needed if you intend 
        to go up to 22 kHz.
    fbi : list or numpy array, optional
        Custom band center frequencies in Hz. If None, a default array is used.
    zetai : float, optional
        Filter bandwidth in octaves. Default is 1.0 (meaning one octave 
        above and below each center frequency).

    Returns
    -------
    rt60 : numpy.ndarray
        Reverberation times (seconds) for each band.
    fb : numpy.ndarray
        Center frequencies (Hz) of the bands actually used.
    
    Notes
    -----
    - This function calls an internal `identify_direct_sound_arrival_threshold` 
      to detect the direct sound arrival.
    - The IR is trimmed so that the analysis starts ~20 ms before the 
      identified direct arrival (but won't go before index 0).
    - If a given bandpass filter range is invalid (e.g., > Nyquist or < 0), 
      that band is skipped in the final results.
    - The smoothing, noise floor estimation, and polynomial fit logic 
      are retained from the original code.

    """

    def identify_direct_sound_arrival_threshold(irm_data, threshold):
        """
        Identify the first sample index where irm_data 
        exceeds 'threshold' * peak_amplitude.
        """
        peak_val = np.max(np.abs(irm_data))
        if peak_val == 0:
            raise ValueError("Impulse response is empty or all zeros.")
        
        scale_factor = peak_val * 32768 / 32767
        above_thresh_indices = np.where((np.abs(irm_data) / scale_factor) > threshold)[0]
        if not len(above_thresh_indices):
            raise ValueError("No sample exceeds the specified threshold.")
        return above_thresh_indices[0]

    # If IR has multiple channels, collapse to mono
    ir = np.asarray(ir)
    if ir.ndim > 1:
        ir = np.mean(ir, axis=1)
    
    # Identify direct sound arrival index (with threshold=0.05, as in original code)
    direct_arrival_idx = identify_direct_sound_arrival_threshold(ir, 0.05)

    # Decide how many samples before direct arrival to include (20 ms)
    pre_arrival_samples = int(round(0.02 * fs))
    start_idx = max(direct_arrival_idx - pre_arrival_samples, 0)
    
    irm = ir[start_idx:]
    
    # Default center frequencies if none provided
    if fbi is None:
        # Original code used 125 * 2^( -2.5:0.1:7.6 ), 
        # but note that might produce many bands outside [f_min, f_max].
        fb_full = 125 * 2 ** np.arange(-2.5, 7.6, 0.1)
    else:
        fb_full = np.array(fbi, dtype=float)
    
    # Filter the band centers to be within [f_min, f_max]
    fb = [value for value in fb_full if f_min <= value <= f_max]
    fb = np.array(fb, dtype=float)
    nbands = len(fb)

    # Set the filter order and limit the analysis duration
    order = 2
    btaps = min(len(irm), int(5 * fs))
    irb = np.zeros((btaps, nbands), dtype=float)

    # Bandpass filtering
    from scipy.signal import butter, filtfilt
    nyquist = fs / 2
    for i, fc in enumerate(fb):
        low = (fc * 2 ** (-zetai / 2)) / nyquist
        high = (fc * 2 ** (zetai / 2)) / nyquist
        # Only apply if valid band range
        if 0 < low < high < 1:
            b, a = butter(order, [low, high], btype='bandpass')
            irb[:, i] = filtfilt(b, a, irm[:btaps])
        else:
            # If invalid, we keep zeros in that column
            pass

    # Smoothing the energy envelope
    from scipy.signal import fftconvolve, get_window
    beta_ms = 100  # 100 ms smoothing
    staps = int(round(beta_ms * fs / 1000))
    smooth_filter = get_window("hann", 2 * staps - 1)
    smooth_filter /= np.sum(smooth_filter)

    # sqrt of convolved power to avoid log of zero
    # shape of irb^2 is (btaps, nbands). We convolve each band separately
    ir_power = irb ** 2
    irbs = np.sqrt(np.maximum(
        fftconvolve(ir_power, smooth_filter[:, None], mode='same'), 
        1e-10
    ))

    # Estimate noise floor from last 'eta' ms
    eta_ms = 200
    etaps = int(round(eta_ms * fs / 1000))
    noise_floor = 20 * np.log10(irbs[-etaps:].mean(axis=0))

    # Decay slope estimation
    tau0_ms = 100  # Late field onset time (ms)
    delta1_db = 5  # Additional margin above noise floor
    rt = np.zeros(nbands, dtype=float)

    for i in range(nbands):
        index0 = int(round(tau0_ms * fs / 1000))
        
        # Decay in dB for each sample in the band
        decay_db = 20 * np.log10(irbs[:, i])
        # Find index1 where the decay hits (noise_floor + delta1_db)
        try:
            index1_offset = np.where(decay_db[index0:] < (noise_floor[i] + delta1_db))[0][0]
            index1 = index0 + index1_offset
        except IndexError:
            # If the decay never reached that level, skip
            continue
        
        idx = np.arange(index0, index1)
        if len(idx) < 2:
            # Not enough data to fit
            continue
        
        # Linear fit of decay vs. time in seconds
        time_s = idx / fs
        slope, intercept = np.polyfit(time_s, decay_db[idx], 1)
        
        # RT = -dec / slope  (since slope is negative in a decaying IR)
        if slope == 0:
            # Avoid division by zero if slope is 0
            rt[i] = 0
        else:
            rt[i] = -dec / slope

    # For convenience, print average RT across bands
    valid_rt = rt[rt > 0]
    if len(valid_rt) > 0:
        print(f"Estimated RT ({dec} dB) across valid bands: {np.mean(valid_rt):.2f} s")
    else:
        print("No valid decay times were computed.")

    return rt, fb

def echodensity(ir, fs, wtaps,length,threshold=0.05):
    """
    Compute the echo density (NED) and response energy profile (REP) 
    of an impulse response over time, using a running window.

    Steps:
    1. Identify the direct sound arrival in the IR using a threshold 
       of its normalized amplitude.
    2. Cut a user-defined time segment of the IR starting ~5 ms before
       the direct arrival.
    3. Use a specified window (or window size) to compute:
       - REP: A smoothed energy profile (RMS) over the window.
       - NED: A normalized echo density measure (ratio of samples 
              above the local mean-square threshold).

    Parameters
    ----------
    ir : numpy.ndarray
        The impulse response data (1D or multi-channel). If multi-channel, 
        each column will be treated as a separate channel.
    fs : float
        Sampling rate in Hz.
    wtaps : int or array_like
        - If an integer, specifies the size of the Hann window in samples.
        - If an array-like, treated as the custom window itself.
    length : float
        Number of seconds of the IR to analyze (starting from ~5 ms 
        before the detected direct sound).
    threshold : float, optional
        Amplitude threshold (as a fraction of peak amplitude) 
        for identifying the direct sound arrival. Default is 0.05.

    Returns
    -------
    ned : numpy.ndarray
        Normalized echo density array of shape (N, C), where N is the 
        number of time samples in the analysis window, and C is the 
        number of channels.
    rep : numpy.ndarray
        Response energy profile (RMS) of the IR window, same shape as `ned`.
    t : numpy.ndarray
        Time vector in seconds for the resulting arrays.

    Notes
    -----
    - The direct sound arrival is detected by examining the IR's peak amplitude 
      and locating the first sample above `threshold * peak_amplitude`.
    - The IR is then trimmed to `(start, start + length_in_samples)`, 
      where `start ~ direct_sound_arrival - 5 ms`.
    - `ned` is computed by checking if instantaneous power exceeds the local 
      (windowed) mean-square threshold (`rep[n]^2`) at each time index.
    - `rep` is effectively the local RMS: sqrt of the windowed average of IR^2.
    """

    def identify_direct_sound_arrival_threshold(ir_array, thr):
        """
        Returns the first index where the IR exceeds 'thr * peak_amplitude'.
        """
        ir_array = np.atleast_1d(ir_array)
        peak_val = np.max(np.abs(ir_array))
        if peak_val == 0:
            raise ValueError("Impulse response is empty or all zeros.")
        scaled_ir = np.abs(ir_array) / peak_val
        idx_candidates = np.where(scaled_ir > thr)[0]
        if len(idx_candidates) == 0:
            raise ValueError(f"No sample found above threshold={thr} * peak_amplitude.")
        return idx_candidates[0]

    # If IR has multiple channels, shape = (samples, channels)
    # If IR is 1D, make sure it's (samples, 1)
    ir = np.atleast_2d(ir)
    if ir.shape[0] < ir.shape[1]:
        # Possibly shape was (channels, samples), so transpose
        ir = ir.T

    # Identify direct sound arrival
    direct_index = identify_direct_sound_arrival_threshold(ir, threshold)

    # Subtract 5 ms from direct sound for the starting sample
    start_offset = int(round(0.005 * fs))
    start = max(direct_index - start_offset, 0)
    end = int(round(length * fs)) + start
    ir = ir[start:end, :]

    # Prepare the window
    if np.isscalar(wtaps):
        wtaps = int(wtaps)
        window = get_window('hann', wtaps)
    else:
        # Assume user provided a custom window array
        window = np.asarray(wtaps)
        wtaps = len(window)

    # Normalize the window so it sums to 1
    window = window / np.sum(window)
    half_w = wtaps // 2

    # Dimensions
    ntaps, nchan = ir.shape

    # Prepare outputs
    ned = np.zeros((ntaps, nchan), dtype=float)
    rep = np.zeros((ntaps, nchan), dtype=float)

    # Zero-pad the squared impulse response for windowing at the edges
    # We'll shape it: (ntaps + 2*half_w, nchan)
    ir2_padded = np.vstack([
        np.zeros((half_w, nchan)),
        ir**2,
        np.zeros((half_w, nchan))
    ])

    # Compute NED and REP
    for n in range(ntaps):
        index = slice(n, n + wtaps)

        # Windowed average power => local RMS
        local_power = np.sum(window[:, None] * ir2_padded[index, :], axis=0)
        rep[n, :] = np.sqrt(local_power) 

        # Compare each sample's power to rep[n]^2
        # True => 1.0, False => 0.0, weighted by window => sum => fraction
        threshold_power = (rep[n, :] ** 2)[None, :]  # shape (1, nchan)
        bool_mask = ir2_padded[index, :] > threshold_power
        ned[n, :] = np.sum(window[:, None] * bool_mask, axis=0)

    # Normalize ned by erfc(1 / sqrt(2))
    from scipy.special import erfc
    ned /= erfc(1 / np.sqrt(2))

    # Create time vector
    t = np.arange(ntaps) / fs

    print("NED done!")
    print("Energy Profile done!")

    return ned, rep, t

def a_weighting(frequencies):
    """
    Compute the IEC 61672:2003 A-weighting curve in linear scale 
    for a given array of frequencies.

    Parameters
    ----------
    frequencies : array_like
        Frequencies in Hz.

    Returns
    -------
    numpy.ndarray
        A-weighting in linear scale. To get dB values, do
        20 * log10(result_of_this_function).
    """
    f = frequencies
    # A-weighting formula constants (IEC 61672:2003)
    ra = (12194.0**2 * f**4) / (
        (f**2 + 20.6**2) 
        * (f**2 + 12194.0**2) 
        * np.sqrt((f**2 + 107.7**2) * (f**2 + 737.9**2))
    )

    # Avoid log of zero
    eps = 1e-10
    a_weight_db = 20 * np.log10(ra + eps) + 2.00  # +2 dB per IEC standard
    return 10 ** (a_weight_db / 20.0)  # Convert from dB back to linear scale

def fft_with_a_weighting(ir, fs):
    """
    Compute the A-weighted energy of an impulse response (IR).

    Parameters
    ----------
    ir : numpy.ndarray
        The impulse response data. Can be 1D or multi-channel 
        (shape: (samples,) or (samples, channels)).
    fs : float
        Sampling rate of 'ir' in Hz.

    Returns
    -------
    float
        The total energy of the IR under A-weighting, computed by 
        summing the squared magnitudes of the FFT times the 
        A-weighting curve.
    """
    # If IR is multi-channel, collapse to mono by averaging channels
    ir = np.asarray(ir)
    if ir.ndim > 1:
        ir = np.mean(ir, axis=1)

    # Number of samples
    N = len(ir)
    # Positive frequency bins
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    # FFT magnitude (positive freqs only)
    fft_magnitude = np.abs(np.fft.rfft(ir))

    # A-weighting in linear scale
    a_weight = a_weighting(freqs)

    # Weighted energy = sum of (A-weighted magnitude)^2
    weighted_energy = np.sum((fft_magnitude * a_weight) ** 2)
    return weighted_energy

def calculate_relative_spl_a_weighted(ir, fs, ref_ir, ref_fs):
    """
    Calculate the A-weighted sound pressure level (SPL) of 'ir' 
    relative to a reference impulse response 'ref_ir'.

    Parameters
    ----------
    ir : numpy.ndarray
        The test impulse response (1D or multi-channel).
    fs : float
        Sampling rate for 'ir'.
    ref_ir : numpy.ndarray
        Reference impulse response (1D or multi-channel).
    ref_fs : float
        Sampling rate for 'ref_ir'.

    Returns
    -------
    float
        The relative A-weighted SPL in decibels (dB). 

    Notes
    -----
    - If you only have absolute values (no reference), you can just 
      compute 'fft_with_a_weighting' on each IR and compare them 
      manually, e.g., 10*log10(energy1/energy2).
    """
    eps = 1e-10

    # Compute A-weighted energies
    reference_energy = fft_with_a_weighting(ref_ir, ref_fs)
    energy = fft_with_a_weighting(ir, fs)

    # Relative SPL in dB
    rel_spl_db = 10.0 * np.log10((energy / (reference_energy + eps)) + eps)
    print(f"A-weighted SPL (relative): {rel_spl_db:.2f} dB")
    return rel_spl_db

def c80d50(ir, fs, length, threshold=0.9):
    """
    Calculate C80 (in dB) and D50 (in %) from an impulse response.

    Parameters
    ----------
    ir : numpy.ndarray
        The impulse response array, which can be 1D (mono) or 2D (multi-channel).
    fs : float
        Sampling rate in Hz.
    length : float
        The time in seconds up to which the total energy is computed for D50.
    threshold : float, optional
        Threshold (as a fraction of the IR's peak amplitude) for identifying the
        direct sound arrival. Default is 0.9.

    Returns
    -------
    c80_val : float
        C80 value in decibels (dB) for the first channel.
    d50_val : float
        D50 value as a percentage (%) for the first channel.

    Notes
    -----
    - The function internally normalizes the IR so its maximum absolute amplitude 
      is ±1. If you have already normalized your IR externally, you can skip that part.
    - If the IR is multi-channel, C80 and D50 are computed per-channel, but only
      the first channel’s values are printed/returned.
    - C80 is computed as:
          10 * log10( (∑(ir^2) from t0 to t0+80ms) / (∑(ir^2) from t0+80ms to end) )
    - D50 is computed as:
          100 * ( (∑(ir^2) from t0 to t0+50ms) / (∑(ir^2) from t0 to t0+length) ) %
      where t0 is the detected direct sound arrival sample.
    """

    def identify_direct_sound_arrival_threshold(ir_array, thr):
        """
        Identify the first sample index where IR exceeds (thr * peak_amplitude).
        """
        peak_val = np.max(np.abs(ir_array))
        if peak_val == 0:
            raise ValueError("Impulse response is empty or all zeros.")
        
        scale_factor = peak_val * 32768 / 32767
        indices = np.where((np.abs(ir_array) / scale_factor) > thr)[0]
        if len(indices) == 0:
            raise ValueError("No sample found above the given threshold.")
        return indices[0]

    # If IR has multiple channels, shape = (samples, channels).
    # If 1D, we reshape to (samples, 1) to unify the indexing.
    if ir.ndim == 1:
        ir = ir[:, np.newaxis]

    # Normalize entire IR so the peak amplitude is +/-1.
    peak_amp = np.max(np.abs(ir))
    if peak_amp == 0:
        raise ValueError("Impulse response is empty or all zeros.")
    scale_factor_global = peak_amp * 32768 / 32767
    ir_norm = ir / scale_factor_global

    # Convert 'length' from seconds to samples
    end_sample = int(round(length * fs))

    # Find direct sound arrival
    istart = identify_direct_sound_arrival_threshold(ir_norm, threshold)

    # Compute C80
    # - Early energy: from istart to istart+80ms
    # - Late energy: from istart+80ms to the end of the IR
    i80 = istart + int(round(0.08 * fs))
    # Sum of squares across each channel => shape (channels,)
    early_energy = np.sum(np.square(ir_norm[istart:i80]), axis=0)
    late_energy = np.sum(np.square(ir_norm[i80:]), axis=0)
    # Ratio
    c80_ratio = early_energy / late_energy
    # Convert to dB using log10
    c80_db = 10 * np.log10(c80_ratio)

    # Compute D50
    # - Early portion: from istart to istart+50ms
    # - Total portion: from istart to istart+length
    i50 = istart + int(round(0.05 * fs))
    early_energy_50 = np.sum(np.square(ir_norm[istart:i50]), axis=0)
    total_energy = np.sum(np.square(ir_norm[istart:istart+end_sample]), axis=0)
    d50_ratio = early_energy_50 / total_energy
    d50_percent = 100 * d50_ratio

    # Print & return the first channel's values
    c80_val = c80_db[0]
    d50_val = d50_percent[0]
    print(f"C80 in dB: {c80_val:.3f}   D50 in %: {d50_val:.3f}")

    return c80_val, d50_val
