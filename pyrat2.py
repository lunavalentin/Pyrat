import numpy as np # for matrix operations and fast math functions
import glob
import os as os # for operating system path manipulation
import scipy.signal as signal # for signal processing functions
from scipy.signal import butter, filtfilt, fftconvolve, get_window, resample, hilbert, sosfilt, zpk2sos
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
import warnings

# Ensure warnings are visible
warnings.simplefilter('always', UserWarning)

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
    ir = np.atleast_1d(ir)
    peak_val = np.max(np.abs(ir))
    if peak_val == 0:
        raise ValueError("Impulse response is empty or all zeros.")
    
    scale_factor = peak_val * 32768 / 32767
    indices_above_thresh = np.where((np.abs(ir) / scale_factor) > threshold)[0]
    
    if len(indices_above_thresh) == 0:
        raise ValueError(f"No sample found above the given threshold ({threshold}).")
        
    return indices_above_thresh[0]

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

def decaytime(ir, fs, dec, f_min=20, f_max=20000, fbi=None, zetai=1.0, method='schroeder'):
    """
    Compute reverberation time (RT60-like metric) in user-specified frequency bands 
    for a given impulse response.

    Parameters
    ----------
    ir : array_like
        The impulse response data.
    fs : float
        Sample rate in Hz.
    dec : float
        Decay level in dB (e.g., 20, 30, 60).
    f_min : float, optional
        Minimum frequency (Hz).
    f_max : float, optional
        Maximum frequency (Hz).
    fbi : list, optional
        Custom band center frequencies.
    zetai : float, optional
        Filter bandwidth in octaves.
    method : str, optional
        'schroeder' (default): Backward integration (ISO 3382). Best for most cases.
        'envelope': Smoothed energy envelope method (JSA/MATLAB style). 

    Returns
    -------
    rt60 : numpy.ndarray
        Reverberation times (seconds) for each band.
    fb : numpy.ndarray
        Center frequencies (Hz) of the bands used.
    """
    # If IR has multiple channels, collapse to mono
    ir = np.asarray(ir)
    if ir.ndim > 1:
        ir = np.mean(ir, axis=1)
    
    # Identify direct sound arrival index
    try:
        direct_arrival_idx = identify_direct_sound_arrival_threshold(ir, 0.05)
    except ValueError:
        # If detection fails, assume start is near 0 or return error
        direct_arrival_idx = 0

    # Pre-roll for filtering (start 50ms before direct sound if possible)
    pre_arrival_samples = int(round(0.05 * fs)) 
    start_idx_processing = max(direct_arrival_idx - pre_arrival_samples, 0)
    
    irm = ir[start_idx_processing:]
    
    # Define frequency bands
    if fbi is None:
        fb_full = 125 * 2 ** np.arange(-2.5, 7.6, 0.1)
    else:
        fb_full = np.array(fbi, dtype=float)
    
    fb = np.array([v for v in fb_full if f_min <= v <= f_max], dtype=float)
    nbands = len(fb)

    # 4th order Butterworth (order=2 passed to butter -> 2 poles * 2 passes = 4 poles effectively? 
    # Actually butter(N) gives 2Nth order usually? No, butter(N) gives Nth order. 
    # filtfilt doubles expected stopband attenuation. 
    # JSA code uses order=2 with filtfilt.
    order = 2
    
    btaps = len(irm)
    irb = np.zeros((btaps, nbands), dtype=float)

    nyquist = fs / 2
    for i, fc in enumerate(fb):
        low = (fc * 2 ** (-zetai / 2)) / nyquist
        high = (fc * 2 ** (zetai / 2)) / nyquist
        if 0 < low < high < 1:
            try:
                b, a = butter(order, [low, high], btype='bandpass')
                irb[:, i] = filtfilt(b, a, irm)
            except Exception:
                pass # Filter error (e.g. unstable)

    rt = np.zeros(nbands, dtype=float)
    
    # Processing per band
    for i in range(nbands):
        # 1. Squared Envelope
        h2 = irb[:, i] ** 2
        
        # Noise floor estimation (last 10% or 200ms)
        noise_len = max(int(0.1 * btaps), int(0.2 * fs))
        if noise_len >= btaps: noise_len = btaps // 2
        
        noise_floor_energy = np.mean(h2[-noise_len:]) if noise_len < btaps else 1e-12
        noise_floor_db = 10 * np.log10(noise_floor_energy + 1e-15)
        
        # Peak energy (to check SNR)
        peak_energy = np.max(h2)
        peak_db = 10 * np.log10(peak_energy + 1e-15)
        
        # SNR Check
        snr_band = peak_db - noise_floor_db
        if snr_band < (dec + 10): 
            # Not enough dynamic range for requested decay
            # We could try to fallback to a smaller decay and extrapolate, but let's be strict or return 0
            # Ideally: if requested 60dB but only have 40dB SNR, we can't reliably give T60.
            # But we can calculate T20 and * 3 if user allows.
            # For now, let's just mark as invalid if SNR is very poor (< 15dB)
            if snr_band < 15:
                rt[i] = 0
                continue

        y_decay = None
        
        if method == 'schroeder':
            # --- Schroeder Backward Integration ---
            # "Intersection" truncation: integrate from where signal meets noise
            # Find last point > noise_floor + 10dB (or similar margin)
            # This is robust for noisy tails.
            
            # Smoothing for endpoint detection
            win_s = int(0.05 * fs)
            smoother = np.ones(win_s)/win_s
            h2_smooth = fftconvolve(h2, smoother, mode='same')
            
            # Find truncation point (intersection with noise + e.g. 5-10dB)
            # Working in dB is often easier
            h2_db = 10 * np.log10(h2_smooth + 1e-15)
            
            # Find last sample > noise_floor_db + 5 dB
            valid_indices = np.where(h2_db > (noise_floor_db + 5))[0]
            if len(valid_indices) == 0:
                trunc_idx = btaps - 1 
            else:
                trunc_idx = valid_indices[-1]
                
            # Integrate backwards from truncation point
            schroeder_curve = np.flip(np.cumsum(np.flip(h2[:trunc_idx] - noise_floor_energy)))
            
            # Avoid negatives/zeros in log
            schroeder_curve = np.maximum(schroeder_curve, 1e-15)
            # Normalize
            schroeder_curve /= np.max(schroeder_curve)
            
            y_decay = 10 * np.log10(schroeder_curve)
            
            # Time axis for this curve matches irb[:trunc_idx]
            # But the start of the decay is the global peak of the IR (direct sound)
            # We need to offset our linear fit relative to the peak of the Schroeder curve?
            # Usually we just take 0dB as t=0 for the fit.
            t_axis = np.arange(len(y_decay)) / fs
            
        else: # method == 'envelope'
            # --- Envelope Method (JSA / MATLAB) ---
            # Smoothing window (beta=100ms)
            beta_ms = 100
            staps = int(round(beta_ms * fs / 1000))
            smooth_win = get_window('hann', 2 * staps - 1)
            smooth_win /= np.sum(smooth_win)
            
            h_env = np.sqrt(np.maximum(
                fftconvolve(h2, smooth_win, mode='same'), 1e-15
            ))
            
            y_decay = 20 * np.log10(h_env) # 20log of amplitude envelope = 10log of energy
            t_axis = np.arange(len(y_decay)) / fs
            
            # JSA logic: fit starts at tau0=100ms
            # Ends at noise_floor + 12dB
            # We'll handle this in the generic fitting block below using 'start_db' logic.
            
        
        # --- Linear Fit Logic (Common) ---
        # Determine dB ranges
        if dec == 10: # EDT
            fit_start_db = 0; fit_end_db = -10
        elif dec == 20: # T20
            fit_start_db = -5; fit_end_db = -25
        elif dec == 30: # T30
            fit_start_db = -5; fit_end_db = -35
        else: # T60 requested
            # Try to get -5 to -65
            fit_start_db = -5; fit_end_db = -65
            
        # Safeguard: if SNR is low, we might not reach fit_end_db.
        # Check if we assume linear decay, does our y_decay go low enough?
        min_y = np.min(y_decay)
        if min_y > fit_end_db + 5:
           # Signal doesn't decay enough for this measure
           rt[i] = 0
           continue

        try:
            # Find indices for start and end levels
            # We want the FIRST time it crosses start_db (after peak)
            # And the FIRST time it crosses end_db
            
            # For Schroeder: curve is monotonic decreasing.
            # For Envelope: might wobble. JSA code uses fixed start time (100ms).
            # We'll use level-based detection for consistency with ISO.
            
            # Find peak index first (t=0)
            peak_idx = np.argmax(y_decay)
            
            # Search after peak
            y_search = y_decay[peak_idx:]
            
            idx_start_rel = np.where(y_search <= fit_start_db)[0]
            if len(idx_start_rel) == 0: continue
            idx_start = peak_idx + idx_start_rel[0]
            
            idx_end_rel = np.where(y_search <= fit_end_db)[0]
            if len(idx_end_rel) == 0: continue
            idx_end = peak_idx + idx_end_rel[0]
            
            if idx_end <= idx_start + 2: continue # Too few points
            
            # Regression
            y_segment = y_decay[idx_start:idx_end]
            t_segment = t_axis[idx_start:idx_end]
            
            slope, intercept = stats.linregress(t_segment, y_segment)[:2]
            
            if slope >= 0:
                rt[i] = 0
            else:
                # Calculate T60
                # Slope is dB/sec. T60 = 60 / |slope|
                # The slope represents the decay rate of the line.
                # If we fitted -5 to -25 (20dB drop), the slope tells us how fast 
                # it drops. e.g. -100 dB/s.
                # RT60 = 60 / 100 = 0.6s.
                # So it's always -60 / slope, regardless of the segment used.
                
                val = -60.0 / slope
                
                # Sanity Check: RT60 shouldn't be absurdly larger than the file duration
                max_valid_rt = (len(ir) / fs) * 1.5
                if val > max_valid_rt or val < 0:
                    rt[i] = 0
                else:
                    rt[i] = val
                
        except Exception:
            rt[i] = 0

    valid_rt = rt[rt > 0]
    if len(valid_rt) > 0:
        print(f"Estimated RT ({method}, {dec} dB basis) - mean valid: {np.mean(valid_rt):.2f} s")
        
    return rt, fb

def echodensity(ir, fs, wtaps, length, threshold=0.05, align=True):
    """
    Compute the echo density (NED) and response energy profile (REP) 
    of an impulse response over time, using a running window.

    Steps:
    1. (Optional) Identify the direct sound arrival in the IR using a threshold 
       of its normalized amplitude.
    2. (Optional) Cut a user-defined time segment of the IR starting ~5 ms before
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
    align : bool, optional
        If True (default), automatically detects direct sound and crops the IR.
        If False, processes the IR as-is (up to 'length' seconds).

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

    # If IR has multiple channels, shape = (samples, channels)
    # If IR is 1D, make sure it's (samples, 1)
    ir = np.atleast_2d(ir)
    if ir.shape[0] < ir.shape[1]:
        # Possibly shape was (channels, samples), so transpose
        ir = ir.T

    if align:
        # Identify direct sound arrival
        direct_index = identify_direct_sound_arrival_threshold(ir, threshold)

        # Subtract 5 ms from direct sound for the starting sample
        start_offset = int(round(0.005 * fs))
        start = max(direct_index - start_offset, 0)
    else:
        start = 0

    end = int(round(length * fs)) + start
    # Ensure end doesn't exceed length
    if end > ir.shape[0]:
        end = ir.shape[0]
        
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

def generate_directory_report(input_folder, output_folder, file_pattern='*.wav'):
    """
    Scans a folder for WAV files, calculates acoustic metrics (RT60, C80, D50, SNR, Echo Density),
    generates waveform/spectrogram/NED plots, and compiles a Markdown report.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing .wav files.
    output_folder : str
        Path to the folder where reports and plots will be saved.
    file_pattern : str, optional
        Glob pattern for finding files (default '*.wav').
    """
    if not os.path.isdir(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plots_folder = os.path.join(output_folder, "plots")
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    wav_files = glob.glob(os.path.join(input_folder, file_pattern))
    wav_files.sort()
    
    if not wav_files:
        print(f"No WAV files found in '{input_folder}' matching '{file_pattern}'.")
        return

    results = []
    
    print(f"Found {len(wav_files)} files. Starting processing...")
    
    for wav_file in wav_files:
        filename = os.path.basename(wav_file)
        print(f"Processing {filename}...")
        
        try:
            # 1. Load Audio
            ir, fs = sf.read(wav_file)
            if ir.ndim > 1:
                # Use first channel for consistency in metrics if multi-channel
                # (Some metrics already handle multi-channel, but for reporting we often look at mono or Ch1)
                ir_mono = ir[:, 0]
            else:
                ir_mono = ir
                
            # Metrics container for this file
            file_metrics = {
                'filename': filename,
                'fs': fs,
                'duration_s': len(ir_mono) / fs,
                'error': None
            }
            
            # 2. SNR
            try:
                snr_val = snr_calc(ir_mono, fs)
                file_metrics['snr'] = snr_val
            except Exception as e:
                print(f"  Error calculating SNR: {e}")
                file_metrics['snr'] = None

            # 3. C80 / D50
            try:
                # Use a reasonable length for analysis, e.g., total duration
                c80, d50 = c80d50(ir_mono, fs, file_metrics['duration_s'])
                file_metrics['c80'] = c80
                file_metrics['d50'] = d50
            except Exception as e:
                print(f"  Error calculating C80/D50: {e}")
                file_metrics['c80'] = None
                file_metrics['d50'] = None
            
            # 4. RT60 (decaytime)
            try:
                # Analyze standardized bands 
                fbi = [125, 250, 500, 1000, 2000, 4000, 8000]
                # Use T30 (decay of 30dB) and multiply by 2 for RT60 estimate, 
                # as full 60dB range is rarely available in measurements.
                rt_bands, freqs = decaytime(ir_mono, fs, 30, f_min=100, f_max=10000, fbi=fbi)
                rt_bands = rt_bands * 2 # Convert T30 to RT60
                
                # Store RT60 at 1kHz specifically for summary
                idx_1k = np.argmin(np.abs(freqs - 1000))
                rt60_1k = rt_bands[idx_1k] if len(rt_bands) > idx_1k else None
                file_metrics['rt60_1k'] = rt60_1k
                file_metrics['rt60_bands'] = rt_bands.tolist()
                file_metrics['rt60_freqs'] = freqs.tolist()
            except Exception as e:
                print(f"  Error calculating RT60: {e}")
                file_metrics['rt60_1k'] = None

            # 5. Echo Density
            try:
                # Window size typically ~20-50ms for echo density
                wtaps = int(0.02 * fs) # 20ms window
                # Analyze a chunk, e.g., 1 second or full duration
                analyze_len = min(1.0, file_metrics['duration_s'] - 0.01)
                
                ned, rep, t_ned = echodensity(ir_mono, fs, wtaps, analyze_len, align=True)
                
                # Generate Echo Density Plot
                fig_ned, ax_ned = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                
                # Plot Normalized Echo Density
                ax_ned[0].plot(t_ned, ned)
                ax_ned[0].set_ylabel('NED')
                ax_ned[0].set_title(f'Normalized Echo Density - {filename}')
                ax_ned[0].grid(True)
                
                # Plot Response Energy Profile
                ax_ned[1].plot(t_ned, rep)
                ax_ned[1].set_ylabel('REP (RMS)')
                ax_ned[1].set_xlabel('Time (s)')
                ax_ned[1].grid(True)
                
                ned_plot_path = os.path.join(plots_folder, f"ned_{filename}.png")
                plt.tight_layout()
                plt.savefig(ned_plot_path)
                plt.close(fig_ned)
                
                file_metrics['ned_plot'] = os.path.relpath(ned_plot_path, output_folder)

            except Exception as e:
                print(f"  Error calculating Echo Density: {e}")
                file_metrics['ned_plot'] = None

            # 6. Waveform and Spectrogram Plot
            try:
                fig_spec, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 2]})
                
                # Waveform
                time_vec = np.arange(len(ir_mono)) / fs
                ax1.plot(time_vec, ir_mono)
                ax1.set_ylabel('Amplitude')
                ax1.set_title(f'Waveform & Spectrogram - {filename}')
                ax1.set_xlim([0, file_metrics['duration_s']])
                ax1.grid(True)
                
                # Spectrogram
                Pxx, freqs_spec, t_spec, im = ax2.specgram(
                    ir_mono, NFFT=2048, Fs=fs, noverlap=1024, cmap='inferno'
                )
                ax2.set_ylabel('Frequency (Hz)')
                ax2.set_xlabel('Time (s)')
                ax2.set_yscale('log')
                ax2.set_ylim([20, fs/2])
                
                spec_plot_path = os.path.join(plots_folder, f"spec_{filename}.png")
                plt.tight_layout()
                plt.savefig(spec_plot_path)
                plt.close(fig_spec)
                
                file_metrics['spec_plot'] = os.path.relpath(spec_plot_path, output_folder)
                
            except Exception as e:
                print(f"  Error generating spectrogram: {e}")
                file_metrics['spec_plot'] = None
                
            results.append(file_metrics)
            
        except Exception as e:
            print(f"Critical error processing {filename}: {e}")
            results.append({'filename': filename, 'error': str(e)})

    # Generate Markdown Report
    report_path = os.path.join(output_folder, "report.md")
    with open(report_path, "w") as f:
        f.write(f"# Acoustic Analysis Report\\n")
        f.write(f"**Input Folder:** `{input_folder}`\\n")
        f.write(f"**Date:** {np.datetime64('today')}\\n\\n")
        
        f.write("## Summary Table\\n")
        f.write("| Filename | SNR (dB) | RT60 @ 1kHz (s) | C80 (dB) | D50 (%) |\\n")
        f.write("| --- | --- | --- | --- | --- |\\n")
        
        for res in results:
            if res.get('error'):
                f.write(f"| {res['filename']} | Error | - | - | - |\\n")
                continue
            
            snr_str = f"{res.get('snr', 0):.2f}" if res.get('snr') is not None else "-"
            rt60_str = f"{res.get('rt60_1k', 0):.2f}" if res.get('rt60_1k') is not None else "-"
            c80_str = f"{res.get('c80', 0):.2f}" if res.get('c80') is not None else "-"
            d50_str = f"{res.get('d50', 0):.1f}" if res.get('d50') is not None else "-"
            
            f.write(f"| {res['filename']} | {snr_str} | {rt60_str} | {c80_str} | {d50_str} |\\n")
        
        f.write("\\n## Detailed Analysis\\n")
        for res in results:
            if res.get('error'):
                continue
                
            f.write(f"### {res['filename']}\\n")
            f.write(f"- **Duration:** {res['duration_s']:.2f} s\\n")
            f.write(f"- **SNR:** {res.get('snr', 0):.2f} dB\\n")
            f.write(f"- **RT60 (1kHz):** {res.get('rt60_1k', 0):.2f} s\\n")
            
            if res.get('spec_plot'):
                f.write(f"![Spectrogram]({res['spec_plot']})\\n")
            
            if res.get('ned_plot'):
                f.write(f"![Echo Density]({res['ned_plot']})\\n")
                
            f.write("\\n---\\n")

    print(f"Report generated at: {report_path}")

def generate_sipi_ir(ir, fs, num_copies=1, bands_per_octave=3):
    """
    Generate statistically independent, perceptually identical (SIPI) copies of an Impulse Response.
    
    Methodology: Echo-Density-Matched Sub-band Envelope Imposition.
    1. Analyze the broadband Echo Density (NED) of the original IR.
    2. Generate a sparse 'Velvet Noise' carrier where pulse density matches the measured NED.
    3. Filter this carrier into sub-bands.
    4. Modulate each band by the *smoothed* energy envelope of the original band.
    5. Sum bands to reconstruct.

    Parameters
    ----------
    ir : array_like
        Original impulse response (1D).
    fs : float
        Sample rate.
    num_copies : int, optional
        Number of SIPI copies.
    bands_per_octave : int, optional
        Bandwidth resolution (default 3 for 1/3 octave).

    Returns
    -------
    sipi_irs : list
        List of generated IRs.
    """
    ir = np.asarray(ir)
    if ir.ndim > 1:
        ir = ir[:, 0]
    
    # 1. Analyze Echo Density
    # Use a window of ~20ms for analysis
    wtaps = int(0.02 * fs)
    # Align=True to focus on the dense part and get correct profile
    # But we need the NED profile for the WHOLE file to map it.
    # So we should probably align=False to keep time base, or handle alignment carefully.
    # Let's use align=False to get a NED vector matching input lenth (roughly).
    # But echodensity() crops the start. We need to handle that.
    # Actually, simpler: just run echodensity on the whole zero-padded thing or handle the offset.
    # The existing echodensity function with align=True returns `t` vector relative to crop.
    # Let's use standard echodensity but align=False to map 1:1 if possible, 
    # but `echodensity` implementation does some cropping logic even then?
    # Let's look at `echodensity` again.
    # It calls `identify_direct_sound` and crops.
    # If align=False, it starts from 0?
    
    # We will use the implementation inside pyrat.
    # If we pass align=False, it should process the whole thing.
    # Let's assume we can get a NED vector of same length as IR (or close).
    
    try:
        # We need to ensure we capture the whole duration.
        duration = len(ir) / fs
        # Run echodensity with align=False to map 1:1
        ned, rep, t_ned = echodensity(ir, fs, wtaps, duration, align=False)
        
        # ned might be slightly shorter due to windowing/processing
        # We need to interpolate NED to sample-level for generation
        ned_interp = np.interp(np.arange(len(ir)), np.arange(len(ned)) * (len(ir)/len(ned)), ned[:, 0])
        
    except Exception as e:
        print(f"Error calculating echo density: {e}. Fallback to White Noise.")
        ned_interp = np.ones(len(ir)) # Full density

    # Clip NED to [0, 1]
    ned_interp = np.clip(ned_interp, 1e-4, 1.0)
    
    # Prepare Filters
    nyquist = fs / 2
    if bands_per_octave == 1:
        f_center = 1000 * 2.0 ** np.arange(-6, 6)
    else:
        f_center = 1000 * 2.0 ** (np.arange(-18, 18) / 3.0)

    valid_centers = [f for f in f_center if 20 <= f < nyquist]
    if not valid_centers: return [np.random.randn(len(ir))]

    sipi_irs = []
    print(f"Generating {num_copies} SIPI copies (ED-matched, {len(valid_centers)} bands)...")

    for _ in range(num_copies):
        # 2. Generate Sparse Carrier (Velvet Noise variant)
        # Pulse rate ~ density. 
        # Full density (NED=1) means ~ Gaussian noise ~ 1 pulse per sample (conceptually).
        # Actually, for "Velvet noise", density usually refers to pulses per second.
        # But here NED is a statistical measure. NED=1 -> Gaussian. NED -> 0 -> Sparse.
        # Relation: NED ~ erf( sqrt(density_in_pulses_per_window) ) ? 
        # A simpler heuristic mapping:
        # Probability of a pulse at sample n: P[n] = NED[n]
        # If NED=1, P=1 (pulse every sample -> like white noise). 
        # If NED=0.1, P=0.1.
        # Magnitudes should be random +/- 1.
        
        # Stochastic generation
        probs = np.random.rand(len(ir))
        # Where rand < ned, placing a pulse.
        # But we need to define "max density". White noise has samples at every point.
        # So we can just mask white noise?
        # Sparse Carrier = WhiteNoise * Mask(where rand < NED)
        # But if we mask white noise, the energy drops. We need to normalize later by envelope.
        
        mask = probs < ned_interp
        # Random binary phase +/- 1
        signs = np.where(np.random.rand(len(ir)) > 0.5, 1, -1)
        carrier = np.zeros(len(ir))
        carrier[mask] = signs[mask]
        
        # 3. Sub-band Synthesis
        sipi_accum = np.zeros_like(ir)
        
        for fc in valid_centers:
            # Filter Bands
            factor = 2 ** (1.0 / (2 * bands_per_octave))
            low = fc / factor
            high = fc * factor
            if high >= nyquist: high = nyquist - 1e-5
            if low <= 0: low = 1e-5
            
            sos = butter(2, [low / nyquist, high / nyquist], btype='bandpass', output='sos')
            
            # Original Band
            band_orig = signal.sosfiltfilt(sos, ir)
            
            # Envelope Extraction
            # Use smoothed envelope (RMS in window) rather than Hilbert,
            # because Hilbert is too jagged for creating smooth decay profiles on sparse sparse carriers.
            # Window ~ 20ms
            rms_win = int(0.02 * fs)
            # Use gaussian or hann window
            win = signal.windows.hann(rms_win)
            win /= np.sum(win)
            
            # Energy envelope
            env_sq = signal.fftconvolve(band_orig**2, win, mode='same')
            env = np.sqrt(np.maximum(env_sq, 1e-18))
            
            # Filter Carrier same way (to get temporal smearing of pulses expected in that band)
            band_carrier = signal.sosfiltfilt(sos, carrier)
            
            # Now we impose the envelope.
            # But the 'band_carrier' has its own envelope (due to sparsity and filtering).
            # If we just multiply, we might double-apply the sparsity?
            # No, 'band_carrier' IS the sparse process filtered.
            # We want to force its global energy envelope to match 'env'.
            
            # Extract carrier envelope
            c_env_sq = signal.fftconvolve(band_carrier**2, win, mode='same')
            c_env = np.sqrt(np.maximum(c_env_sq, 1e-18))
            
            # Demodulate carrier (whiten) then Remodulate
            # Avoid divide by zero
            band_sipi = (band_carrier / (c_env + 1e-9)) * env
            
            sipi_accum += band_sipi
            
        sipi_irs.append(sipi_accum)

    return sipi_irs
