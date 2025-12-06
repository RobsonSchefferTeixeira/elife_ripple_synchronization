import os
import numpy as np
import pandas as pd
import math
import sklearn
from joblib import Parallel, delayed
from multiprocessing import Process
from scipy import signal
from scipy import stats as stats
from scipy.io import loadmat
from scipy import signal as sig

# from pcx1_helper_functions import decoding_model_helper_functions as decode

from bs4 import BeautifulSoup
# you need BeautifulSoup and lxml packages installed

def delete_clip_path(svg_file):
    
    with open(svg_file, 'r') as file:
        content = file.read()
    soup = BeautifulSoup(content, 'lxml-xml')
    
    clip_paths = soup.find_all('clipPath')
    
    if clip_paths:
        for clip_path in clip_paths:
            # Remove the clip-path attribute from <clipPath> elements
            clip_path.decompose()
            
        print("ClipPath elements deleted.")
    
        modified_content = str(soup)
        
        # Save modified content to a new file
        with open(svg_file, 'w') as output_file:
            output_file.write(modified_content)
        print("File saved.")

    else:
        print("ClipPath elements not found.")


def caller_saving(inputdict, filename, saving_path):
    os.chdir(saving_path)
    output = open(filename, 'wb')
    np.save(output, inputdict)
    output.close()
    print('File saved.')


def filename_constructor(saving_string, animal_id, dataset, brain_region, ch):
    first_string = saving_string
    animal_id_string = '.' + animal_id
    dataset_string = '.Dataset.' + dataset
    brain_region_string = '.Brain_region.' + brain_region
    electrode_string = '.Electrode.' + str(ch)

    filename_checklist = np.array([first_string, animal_id, dataset, brain_region_string, ch])
    inlcude_this = np.where(filename_checklist != None)[0]

    filename_backbone = [first_string, animal_id_string, dataset_string, brain_region_string, electrode_string]

    filename = ''.join([filename_backbone[i] for i in inlcude_this])

    return filename


def find_matching_indexes(target_timestamps, reference_time_vector,error_threshold = 0.1):
    """
    Find the indexes in a reference time vector that match target timestamps with some maximum error.
    It will find the closest point in reference_time_vector such that
    reference_time_vector[matching_indexes] - target_timestamps < error_threshold

    Parameters:
        target_timestamps (array-like): Timestamps to match in the reference time vector.
        reference_time_vector (array-like): The reference time vector.

    Returns:
        matching_indexes (array): Indexes in the reference time vector that match the target timestamps.
        
        
    """
    if len(target_timestamps) > 0:
        # Find the indexes in the reference time vector that correspond to target timestamps.
        matching_indexes = search_sorted_indices(reference_time_vector, target_timestamps)
        
        # Filter the indexes to retain only those with a matching error less than 100 ms.
        
        # matching_indexes = [idx for idx in matching_indexes if abs(reference_time_vector[idx] - target_timestamps[idx]) < error_threshold]
        I_keep = np.abs((reference_time_vector[matching_indexes]-target_timestamps)) < error_threshold
        matching_indexes = matching_indexes[I_keep]

    else:
        matching_indexes = []
    
    return matching_indexes

''' 
# This is another way to do that. But instead of middle values, we decide which side has smaller error.
def find_timestamps_indexes(timevector, timestamps):
    # Find the indices where timestamps should be inserted in timevector
    left_indices = np.searchsorted(timevector, timestamps, side='left')
    right_indices = np.searchsorted(timevector, timestamps, side='right')
    
    # Calculate absolute differences between timestamps and values at left and right indices
    left_diffs = np.abs(timevector[left_indices] - timestamps)
    right_diffs = np.abs(timevector[right_indices] - timestamps)
    
    # Choose the indices with lower absolute differences
    timestamps_indexes = [left if left_diff <= right_diff else right for left, right, left_diff, right_diff in zip(left_indices, right_indices, left_diffs, right_diffs)]

    return np.array(timestamps_indexes)
'''

def search_sorted_indices(known_array, test_array):
    """
    Search for the indexes in a sorted known array that correspond to values in a test array.

    Parameters:
        known_array (array-like): The sorted array containing known values.
        test_array (array-like): The array containing values to search for in the known array.

    Returns:
        indices (array): Indexes in the known array that correspond to values in the test array.
    """
    # Sort the known array and calculate the middle values between sorted elements.
    sorted_indices = np.argsort(known_array)
    sorted_known_array = known_array[sorted_indices]
    middle_values = sorted_known_array[1:] - np.diff(sorted_known_array.astype('float')) / 2
    
    # Search for the indexes in the middle values that correspond to values in the test array.
    search_results = np.searchsorted(middle_values, test_array)
    indices = sorted_indices[search_results]
    
    return indices

def expand_dimensions(arr):
    if len(arr.shape) == 2:
        return np.expand_dims(arr, axis=1)
    elif len(arr.shape) == 3:
        return arr
    else:
        raise ValueError("Input array should have 2 or 3 dimensions.")
    

def respiration_comodul(respiration,LFP,sampling_rate,PhaseFreqVector,PhaseFreq_BandWidth,AmpFreqVector,AmpFreq_BandWidth):
    # Comodulogram

    # PhaseFreqVector = np.array(range(0,22,2))
    # PhaseFreq_BandWidth = np.array(4)
    # AmpFreqVector = np.array(range(25,205,5))
    # AmpFreq_BandWidth = np.array(10)

    Comodulogram = np.empty([len(PhaseFreqVector),len(AmpFreqVector)])

    
    nbin = 18
    position = np.zeros(nbin)
    winsize = (2*math.pi)/nbin

    for jj in range(0,nbin):
        position[jj] = -math.pi + (jj)*winsize;
    

    PHASES = np.zeros( (len(PhaseFreqVector),len(respiration)))
    # PHASES.shape

    con = 0
    for Pf1 in PhaseFreqVector:
        print(Pf1)
        Pf2 = Pf1 + PhaseFreq_BandWidth
        PhaseFreq = eegfilt(signal.detrend(respiration),sampling_rate,Pf1,Pf2)
        analytic_signal = hilbert(PhaseFreq)
        Phase = np.angle(analytic_signal)

        PHASES[con,:] = Phase
        con = con + 1
    
    con2 = 0
    for Af1 in AmpFreqVector:
        print(Af1)
        
        Af2 = Af1 + AmpFreq_BandWidth
        AmpFreq = eegfilt(signal.detrend(LFP),sampling_rate,Af1,Af2)
        analytic_signal = hilbert(AmpFreq)
        Amp = np.abs(analytic_signal)
    
        con = 0
        for Pf1 in PhaseFreqVector:
            
            Phase = PHASES[con,:]
        
        
            MeanAmp = np.zeros(nbin)
            for ii in range(0,nbin):
                I = np.logical_and(Phase >= position[ii],Phase < position[ii] + winsize)
                MeanAmp[ii] = np.mean(Amp[I])
            

            MI=(np.log(nbin)-(-sum((MeanAmp/sum(MeanAmp))*np.log((MeanAmp/sum(MeanAmp))))))/np.log(nbin);
            Comodulogram[con,con2]=MI;
    
            con = con + 1
    
        con2 = con2 + 1

    return Comodulogram


def variableband_comodul(LFP,sampling_rate,PhaseFreqVector,PhaseFreq_BandWidth,AmpFreqVector):
    # Comodulogram
    import math
    import numpy as np
    from scipy import signal
    

    Comodulogram = np.empty([len(PhaseFreqVector),len(AmpFreqVector)])

    
    nbin = 18
    position = np.zeros(nbin)
    winsize = (2*math.pi)/nbin

    for jj in range(0,nbin):
        position[jj] = -math.pi + (jj)*winsize;
    

    PHASES = np.zeros( (len(PhaseFreqVector),len(LFP)))
    # PHASES.shape

    con = 0
    for Pf1 in PhaseFreqVector:
        #print(Pf1)
        Pf2 = Pf1 + PhaseFreq_BandWidth
        PhaseFreq = eegfilt(signal.detrend(LFP),sampling_rate,Pf1,Pf2)
        analytic_signal = hilbert(PhaseFreq)
        Phase = np.angle(analytic_signal)

        PHASES[con,:] = Phase
        con = con + 1
    
    con2 = 0
    for AmpCenter in AmpFreqVector:
        #print(AmpCenter)
        
        
        con = 0
        for Pf1 in PhaseFreqVector:
            
            Phase = PHASES[con,:]
            CenterPhaseFrequency = Pf1 + PhaseFreq_BandWidth/2
            Af1 = AmpCenter - CenterPhaseFrequency - 2
            Af2 = AmpCenter + CenterPhaseFrequency + 2
            print("Phase Center Frequency: {} ; Amp Center = {}; Lower Amp Freq = {} ; Upper Amp Freq = {}".format(CenterPhaseFrequency, AmpCenter,Af1, Af2))  
            
            AmpFreq = eegfilt(signal.detrend(LFP),sampling_rate,Af1,Af2)
            analytic_signal = hilbert(AmpFreq)
            Amp = np.abs(analytic_signal)
    
        
            MeanAmp = np.zeros(nbin)
            for ii in range(0,nbin):
                I = np.logical_and(Phase >= position[ii],Phase < position[ii] + winsize)
                MeanAmp[ii] = np.mean(Amp[I])
            

            MI=(np.log(nbin)-(-sum((MeanAmp/sum(MeanAmp))*np.log((MeanAmp/sum(MeanAmp))))))/np.log(nbin);
            
            Comodulogram[con,con2]=MI;
    
            con = con + 1
    
        con2 = con2 + 1
        
    return Comodulogram



def tort_comodul(LFP,sampling_rate,PhaseFreqVector,PhaseFreq_BandWidth,AmpFreqVector,AmpFreq_BandWidth):
    # Comodulogram

    # PhaseFreqVector = np.array(range(0,22,2))
    # PhaseFreq_BandWidth = np.array(4)
    # AmpFreqVector = np.array(range(25,205,5))
    # AmpFreq_BandWidth = np.array(10)

    Comodulogram = np.empty([len(PhaseFreqVector),len(AmpFreqVector)])

    
    nbin = 18
    position = np.zeros(nbin)
    winsize = (2*math.pi)/nbin

    for jj in range(0,nbin):
        position[jj] = -math.pi + (jj)*winsize
    

    PHASES = np.zeros( (len(PhaseFreqVector),len(LFP)))
    # PHASES.shape

    con = 0
    for Pf1 in PhaseFreqVector:
        print(Pf1)
        Pf2 = Pf1 + PhaseFreq_BandWidth
        PhaseFreq = eegfilt(signal.detrend(LFP),sampling_rate,Pf1,Pf2)
        analytic_signal = hilbert(PhaseFreq)
        Phase = np.angle(analytic_signal)

        PHASES[con,:] = Phase
        con = con + 1
    
    con2 = 0
    for Af1 in AmpFreqVector:
        print(Af1)
        
        Af2 = Af1 + AmpFreq_BandWidth
        AmpFreq = eegfilt(signal.detrend(LFP),sampling_rate,Af1,Af2)
        analytic_signal = hilbert(AmpFreq)
        Amp = np.abs(analytic_signal)
    
        con = 0
        for Pf1 in PhaseFreqVector:
            
            Phase = PHASES[con,:]
        
        
            MeanAmp = np.zeros(nbin)
            for ii in range(0,nbin):
                I = np.logical_and(Phase >= position[ii],Phase < position[ii] + winsize)
                MeanAmp[ii] = np.mean(Amp[I])
            

            MI=(np.log(nbin)-(-sum((MeanAmp/sum(MeanAmp))*np.log((MeanAmp/sum(MeanAmp))))))/np.log(nbin);
            Comodulogram[con,con2]=MI;
    
            con = con + 1
    
        con2 = con2 + 1

    return Comodulogram

def mean_amp_distribution(phase_series,amplitude_series,n_bin=18):

    position = np.zeros(n_bin)
    winsize = (2*math.pi)/n_bin
    for jj in range(0,n_bin):
        position[jj] = -math.pi + jj*winsize

    mean_amp = np.zeros(n_bin)
    for ii in range(0,n_bin):
        phase_idx = np.logical_and(phase_series >= position[ii],phase_series < position[ii] + winsize)
        mean_amp[ii] = np.nanmean(amplitude_series[phase_idx])

    center_phase_bins = position+winsize/2
    return mean_amp,center_phase_bins


def mod_index(mean_amp,n_bin):
    modulation_index = (np.log(n_bin)-(-np.nansum((mean_amp/np.nansum(mean_amp))*np.log((mean_amp/np.nansum(mean_amp))))))/np.log(n_bin);
    return modulation_index



def next_power_of_2(n):
    """
    Return the next power of 2 greater than or equal to 'n'.

    Parameters
    ----------
    n : int
        Input number.

    Returns
    -------
    next_power : int
        Next power of 2 greater than or equal to 'n'.

    Notes
    -----
    This function calculates the next power of 2 that is greater than or equal to the input 'n'.
    It finds the next power of 2 by performing bitwise operations, shifting bits to determine the nearest power of 2.
    """
    n -= 1                 # short for "n = n - 1"
    shift = 1
    while (n+1) & n:       # the operator "&" is a bitwise operator: it compares every bit of (n+1) and n, and returns those bits that are present in both
        n |= n >> shift    
        shift <<= 1        # the operator "<<" means "left bitwise shift": this comes down to "shift = shift**2"
    return n + 1

# def my_wavelet(lfp,sampling_rate,freqvector,norm=True,stand=True):
#     import numpy as np
    
#     dt = 1/sampling_rate
#     Fc = 0.8125
#     scales = Fc/(freqvector*dt)
#     TSD = np.empty([len(scales),len(lfp)])

#     count = 0
#     for scale in scales:
#         step = 1/scale
#         time = np.arange(-4.0,4.0,step)
#         MorletWav = np.exp(-(time**2)/2)*np.cos(5*time)
        
#         if norm == False:
#             cmor = MorletWav
#         else:
#             cmor = MorletWav/(.5*np.nansum(np.abs(MorletWav)))
        
#         convolution = hilbert(np.convolve(lfp,cmor,'same'))
        
#         if stand == False:
#             TSD[count,:] = np.abs(convolution)
#         else:
#             TSD[count,:] = np.abs(convolution)**2/(np.std(convolution)**2)

#         count = count + 1
#     return TSD

def notch_filter(lfp, fs, notch_freq, quality_factor=30):
    """
    Apply a notch (band-stop) filter to remove a specific frequency from the LFP signal.

    Parameters
    ----------
    lfp : array-like
        Input signal (e.g., LFP).
    fs : float
        Sampling frequency of the input signal in Hz.
    notch_freq : float
        Frequency to be removed (e.g., 50 or 60 Hz for line noise).
    quality_factor : float, optional
        Quality factor (Q) that determines the bandwidth of the notch (default is 30).

    Returns
    -------
    filtered : array-like
        Notch-filtered signal.
    """
    b_notch, a_notch = sig.iirnotch(w0=notch_freq, Q=quality_factor, fs=fs)
    filtered = sig.filtfilt(b_notch, a_notch, lfp)
    return filtered

def eegfilt(LFP, fs, lowcut, highcut, axis=-1, order=3, notch_freq=None, notch_bandwidth=1):
    """
    Apply a bandpass, lowpass, highpass, or optional notch filter to the input signal (LFP),
    using second-order sections for numerical stability.

    Parameters
    ----------
    LFP : array-like
        Input signal (e.g., LFP).
    fs : float
        Sampling frequency of the input signal in Hz.
    lowcut : float
        Low cutoff frequency (Hz). Set to 0 for lowpass only.
    highcut : float
        High cutoff frequency (Hz). Set to 0 for highpass only.
    axis : int, optional
        Axis along which to apply the filter (default is -1).
    order : int, optional
        Order of the Butterworth filter (default is 3).
    notch_freq : float, optional
        Frequency to apply a notch filter at (default is None = no notch).
    notch_bandwidth : float, optional
        Bandwidth around the notch frequency in Hz (default is 1 Hz).

    Returns
    -------
    filtered : array-like
        Filtered signal.
    """

    nyq = 0.5 * fs
    low = lowcut / nyq if lowcut else 0
    high = highcut / nyq if highcut else 0

    # Apply notch filter if specified
    if notch_freq is not None:
        q = notch_freq / notch_bandwidth
        b_notch, a_notch = sig.iirnotch(w0=notch_freq / nyq, Q=q)
        LFP = sig.filtfilt(b_notch, a_notch, LFP, axis=axis)

    # Apply main filter (bandpass, lowpass, or highpass)
    if low == 0 and high > 0:
        sos = sig.butter(order, high, btype='low', output='sos')
    elif high == 0 and low > 0:
        sos = sig.butter(order, low, btype='high', output='sos')
    elif 0 < low < high < 1:
        sos = sig.butter(order, [low, high], btype='band', output='sos')
    else:
        raise ValueError("Invalid lowcut/highcut combination. Ensure 0 ≤ lowcut < highcut < fs/2.")

    filtered = sig.sosfiltfilt(sos, LFP, axis=axis)
    return filtered

'''
def hilbert(lfp):
    hilbert = signal.hilbert(lfp,next_power_of_2(len(lfp)))
    hilbert = hilbert[range(len(lfp))]
    
    return hilbert
'''

def complex_pca(data, n_components):

    """
    Perform Principal Component Analysis (PCA) on complex-valued data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data matrix of shape (samples, features) or (features,) for 1D data.
    n_components : int
        Number of components to keep.

    Returns
    -------
    projected_data : numpy.ndarray
        Projected data onto the selected principal components.
    selected_eigenvectors : numpy.ndarray
        Top 'n_components' eigenvectors representing principal components.
    reconstructed_data : numpy.ndarray
        Reconstructed data using the selected principal components.

    Raises
    ------
    ValueError
        If the input data does not have a valid shape (2D for samples, features).

    Notes
    -----
    This function computes PCA on complex-valued data, producing the projected data,
    selected eigenvectors (principal components), and the reconstructed data using
    the selected principal components.
    """
    # Ensure the input data is in the correct shape: (samples, features)
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0).T  # Convert 1D array to 2D column vector
    elif data.ndim != 2:
        raise ValueError("Input data should be a 2D array (samples, features)")

    # Calculate mean of the complex data
    mean_data = np.nanmean(data, axis=0, keepdims=True)

    # Center the data
    centered_data = data - mean_data

    # Calculate covariance matrix
    cov_matrix = np.cov(centered_data, rowvar=False)

    # Compute eigenvectors and eigenvalues of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'n_components' eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]

    # Project the centered data onto the selected eigenvectors
    projected_data = np.dot(centered_data, selected_eigenvectors)

    # Reconstruct the data using the selected eigenvectors
    reconstructed_data = np.dot(projected_data, selected_eigenvectors.T) + mean_data

    return projected_data, selected_eigenvectors, reconstructed_data



def hilbert(lfp):

    """
    Compute the Hilbert transform of the input signal(s).

    Parameters
    ----------
    lfp : numpy.ndarray
        Input signal. If 1D, treated as a single-channel signal. If 2D, each row represents a channel.

    Returns
    -------
    hilbert_result : numpy.ndarray
        Complex-valued array containing the Hilbert transform of the input signal(s). 
        If the input was 1D, the output is 1D. If the input was 2D, the output preserves the original shape.

    Raises
    ------
    ValueError
        If the input array is not 1D or 2D.

    Notes
    -----
    The Hilbert transform creates an analytic signal by adding a 90-degree phase-shifted version of the input signal.
    This function handles both single-channel and multi-channel signals, applying the transform independently to each channel.
    """
    
    if lfp.ndim == 2:
        n_channels, n_timepoints = lfp.shape
    elif lfp.ndim == 1:
        n_channels = 1
        n_timepoints = len(lfp)
        lfp = lfp.reshape((n_channels, n_timepoints))
    else:
        raise ValueError("Input array should be 1D or 2D")

    hilbert_result = np.zeros_like(lfp, dtype=np.complex128)  # Initialize complex-valued array

    for channel_idx in range(n_channels):
        channel_data = lfp[channel_idx]  # Extract data for each channel
        hilbert_channel = signal.hilbert(channel_data, next_power_of_2(len(channel_data)))  # Apply Hilbert transform
        hilbert_result[channel_idx] = hilbert_channel[range(len(channel_data))]  # Store the result for each channel

    return hilbert_result.squeeze() if n_channels == 1 else hilbert_result




def get_burst_timestamps(spike_timestamps, first_isi_threshold=80, isi_burst_threshold=160):
    """
    Extract burst timestamps from a spike train based on specified inter-spike interval (ISI) thresholds.

    Parameters:
        spike_timestamps (numpy.ndarray): An array of spike timestamps (sorted in ascending order).
        first_isi_threshold (float, optional): The ISI threshold for the first spike in a burst (default is 80).
        isi_burst_threshold (float, optional): The ISI threshold for subsequent spikes within a burst (default is 160).

    Returns:
        burst_timestamps (numpy.ndarray): An array of spike timestamps corresponding to burst times.
    """
    isi_spike_timestamps = np.diff(spike_timestamps)
    burst_id_timestamps = np.zeros(spike_timestamps.shape[0])

    burst_count = 0
    current_spike_index = 0
    while current_spike_index < isi_spike_timestamps.shape[0]:
        if isi_spike_timestamps[current_spike_index] < first_isi_threshold:
            inside_burst = True
            burst_count += 1
            burst_id_timestamps[current_spike_index:current_spike_index + 2] = burst_count
            while inside_burst:
                current_spike_index += 1
                if current_spike_index == isi_spike_timestamps.shape[0]:
                    break

                if isi_spike_timestamps[current_spike_index] < isi_burst_threshold:
                    burst_id_timestamps[current_spike_index:current_spike_index + 2] = 1
                else:
                    inside_burst = False
        else:
            current_spike_index += 1

    return spike_timestamps[np.where(burst_id_timestamps)[0]]



def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

def smooth(x,window_len=11,window='hanning'):
    
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def gaussian_smooth_1d(input_data, sigma_points):
    """
    Perform 1D Gaussian smoothing on input data.
    Notice that when using it for time series, sigma_points are set in poins, not time.
    In order to set the correct amount of points that correspond to ms, for instance,
    one should use it like this: sigma_points = (s/1000)*sampling_rate
    where s in the standard deviation in ms.

    Parameters:
        input_data (numpy.ndarray): The 1D input data to be smoothed.
        sigma_points (float): The standard deviation of the Gaussian kernel in data points.

    Returns:
        smoothed_data (numpy.ndarray): The smoothed 1D data.
    """
    # Generate a 1D Gaussian kernel.
    gaussian_kernel_1d = generate_1d_gaussian_kernel(sigma_points)

    # Convolve the input data with the Gaussian kernel.
    if input_data.ndim == 1:
        input_data = np.expand_dims(input_data,0)
    # smoothed_data = sig.convolve(input_data, gaussian_kernel_1d, mode='same')
    smoothed_data = np.apply_along_axis(lambda x: sig.convolve(x, gaussian_kernel_1d, mode='same'), axis=1, arr=input_data)

    return np.squeeze(smoothed_data)


def generate_1d_gaussian_kernel(sigma):
    """
    Generate a 1D Gaussian kernel with a specified standard deviation.

    Parameters:
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        gaussian_kernel (numpy.ndarray): The 1D Gaussian kernel.
    """
    x_values = np.arange(-3.0 * sigma, 3.0 * sigma + 1.0)
    constant = 1 / (np.sqrt(2 * math.pi) * sigma)
    gaussian_kernel = constant * np.exp(-((x_values**2) / (2 * (sigma**2))))

    return gaussian_kernel


def gaussian_smooth_2d(input_matrix, sigma_points):
    """
    Perform 2D Gaussian smoothing on input data.

    Parameters:
        input_matrix (numpy.ndarray): The 2D input matrix to be smoothed.
        sigma_points (float): The standard deviation of the 2D Gaussian kernel in data points.

    Returns:
        smoothed_matrix (numpy.ndarray): The smoothed 2D data.
    """
    # Generate a 2D Gaussian kernel.
    gaussian_kernel_2d = generate_2d_gaussian_kernel(sigma_points)

    # Convolve the input matrix with the 2D Gaussian kernel.
    smoothed_matrix = sig.convolve2d(input_matrix, gaussian_kernel_2d, mode='same')

    return smoothed_matrix

def generate_2d_gaussian_kernel(sigma):
    """
    Generate a 2D Gaussian kernel with a specified standard deviation.

    Parameters:
        sigma (float): The standard deviation of the 2D Gaussian kernel.

    Returns:
        gaussian_kernel (numpy.ndarray): The 2D Gaussian kernel.
    """
    x_values = np.arange(-3.0 * sigma, 3.0 * sigma + 1.0)
    y_values = np.arange(-3.0 * sigma, 3.0 * sigma + 1.0)
    
    gaussian_kernel = np.zeros([y_values.shape[0], x_values.shape[0]])
    
    for x_count, x_val in enumerate(x_values):
        for y_count, y_val in enumerate(y_values):
            gaussian_kernel[y_count, x_count] = np.exp(-((x_val**2 + y_val**2) / (2 * (sigma**2))))

    return gaussian_kernel



def get_mod_index(phase_series, amp_series, nbin=18, starting_phase = -math.pi):

    position = np.zeros(nbin)
    winsize = (2*math.pi)/nbin

    for jj in range(0,nbin):
        position[jj] = starting_phase + (jj)*winsize
        
    mean_amp = np.zeros(nbin)
    for ii in range(nbin):
        I = np.logical_and(phase_series >= position[ii], phase_series < position[ii] + winsize)
        mean_amp[ii] = np.mean(amp_series[I])

    mod_index = (np.log(nbin) - (-np.nansum((mean_amp / np.nansum(mean_amp)) * np.log((mean_amp / np.nansum(mean_amp)))))) / np.log(nbin)
    return mod_index,mean_amp


def xcorr_maxlag(x, y,normed=False, maxlag=1000):
    
    # y is the reference. A peak after 0 means that x fires after y
    # one can test using this:
    # x = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    # y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    # print x
    # print y
    # xcorr_maxlag(x, y, maxlag=5)


    import numpy as np
    xl = x.size
    yl = y.size

    c = np.zeros(2*maxlag + 1)



    for i in range(maxlag+1):
        tmp = np.correlate(x[0:min(xl, yl-i)], y[i:i+min(xl, yl-i)])
        # tmp = sig.correlate(x[0:min(xl, yl-i)], y[i:i+min(xl, yl-i)],mode='valid',method='auto')
        c[maxlag-i] = tmp[0]
        tmp = np.correlate(x[i:i+min(xl-i, yl)], y[0:min(xl-i, yl)])
        # tmp = sig.correlate(x[i:i+min(xl-i, yl)], y[0:min(xl-i, yl)],mode='valid',method='auto')        
        c[maxlag+i] = tmp[0]
    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    return c

def xcorr_maxlag2(x, y, maxlag=1000):
    
    # y is the reference. A peak after 0 means that x fires after y
    # one can test using this:
    # x = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0])
    # y = np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    # print x
    # print y
    # xcorr_maxlag(x, y, maxlag=5)


    import numpy as np
    xl = x.size
    yl = y.size

    c = np.zeros(2*maxlag + 1)

    for i in range(maxlag+1):
        tmp = np.corrcoef(x[0:min(xl, yl-i)], y[i:i+min(xl, yl-i)])[0,1]
        c[maxlag-i] = tmp
        tmp = np.corrcoef(x[i:i+min(xl-i, yl)], y[0:min(xl-i, yl)])[0,1]
        c[maxlag+i] = tmp

    return c


def autocorr(x, lags):
    xcorr = np.correlate(x - x.mean(), x - x.mean(), 'full')  # Compute the autocorrelation
    xcorr = xcorr[xcorr.size//2:] / xcorr.max()               # Convert to correlation coefficients
    return xcorr[:lags+1]                                     # Return only requested lags







def bin_discrete_events(event_timestamps, time_vector, sample_rate, bin_duration=0.1, slide_this_much=0.1):
    """
    Bins discrete events based on timestamps and a time vector.

    Parameters:
        event_timestamps (array-like): Timestamps of discrete events such that time_vector[event_timestamps]
        time_vector (array-like): Time vector associated with the events.
        sample_rate (float): Sample rate of the time vector (samples per second).
        bin_duration (float, optional): Duration of each time bin in seconds (default is 0.1 seconds).
        slide_this_much (float, optional): Sliding interval between bins in seconds (default is 0.1 seconds).

    Returns:
        binned_event_counts (array): Binned counts of events for each time bin.
        binned_time_vector (array): Binned time vectors corresponding to each time bin.
    """
    # Convert bin_duration and slide_this_much from seconds to samples using the provided sample rate (sample_rate).
    bin_duration_samples = bin_duration * sample_rate
    slide_this_much_samples = slide_this_much * sample_rate

    # Calculate the length of the time_vector.
    time_vector_length = time_vector.shape[0]

    # Calculate the number of bins based on bin duration and slide_this_much.
    num_bins = np.floor((time_vector_length - bin_duration_samples) / slide_this_much_samples + 1).astype(int)

    # Initialize empty lists to store binned event counts and binned time vectors.
    binned_event_counts = []
    binned_time_vector = []

    # Iterate over each bin.
    for bin_index in range(num_bins):
        # Calculate the indices for the current bin's time window.
        if (bin_index * slide_this_much_samples + bin_duration_samples + 1) > time_vector.shape[0]:
            # Ensure that the last bin does not extend beyond the length of the time_vector.
            bin_indices = np.arange(bin_index * slide_this_much_samples, time_vector.shape[0]).astype(int)
        else:
            bin_indices = np.arange(bin_index * slide_this_much_samples, bin_index * slide_this_much_samples + bin_duration_samples).astype(int)

        # Count the number of events that fall within the current time window and append to binned_event_counts.
        binned_event_counts.append(np.nansum(np.in1d(event_timestamps, bin_indices)))

        # Calculate the mean time value within the current time window and append to binned_time_vector.
        binned_time_vector.append(np.nanmean(time_vector[bin_indices]))

    # Convert the binned event counts and time vectors into numpy arrays.
    binned_event_counts = np.array(binned_event_counts).T
    binned_time_vector = np.array(binned_time_vector)

    # Return the binned event counts and time vectors.
    return binned_event_counts, binned_time_vector




def mean_vector_length(phases, weights=None):
    """
    Calculate the mean vector length and angle from weighted phase values.

    Parameters:
        phases (numpy.ndarray): An array of phase angles in radians.
        weights (numpy.ndarray, optional): An array of weights corresponding to each phase angle.

    Returns:
        mean_vector_length (float): The mean vector length.
        mean_angle (float): The mean angle in radians.
    """
    if weights is None:
        weights = np.ones(len(phases))
    else:
        if len(phases) != len(weights):
            raise ValueError("Phases and weights must have the same length.")

    weights = weights / np.nansum(weights)

    # Calculate the weighted sum of complex unit vectors.
    complex_vectors = np.exp(1j * phases)
    weighted_vectors = weights * complex_vectors
    mean_vector = np.nansum(weighted_vectors)

    # Calculate the mean vector length and angle from the weighted components.
    mvl = np.abs(mean_vector)
    mean_angle = np.angle(mean_vector)

    return mvl, mean_angle

    
def circular_mean_real(phase_angles, weights = None):
    """
    Calculate the circular mean of phase angles with weighted values.

    Parameters:
        phase_angles (numpy.ndarray): Array of phase angles in radians.
        weights (numpy.ndarray, optional): Array of weights for each phase angle.

    Returns:
        mean_angle (float): Circular mean angle in radians.
    """

    if weights is None:
        weights = np.ones(len(phase_angles))
    else:
        if len(phase_angles) != len(weights):
            raise ValueError("Phases and weights must have the same length.")
    
    # Calculate the weighted sum of sine and cosine components.
    sum_sin = np.nansum(weights * np.sin(phase_angles))
    sum_cos = np.nansum(weights * np.cos(phase_angles))

    # Calculate the mean angle from the weighted components.
    mean_angle = np.arctan2(sum_sin, sum_cos)

    # Ensure the mean angle is in the [0, 2*pi] range.
    mean_angle = (mean_angle + 2 * np.pi) % (2 * np.pi)

    return mean_angle


def circular_mean_complex(samples, weights=None, high=2*np.pi, low=0, axis=None, nan_policy='propagate', keepdims=False):
    """
    Compute the (weighted) circular mean of angle observations using complex numbers.

    Parameters:
    - samples: array_like
        Input array of angle observations.
    - weights: array_like, optional
        Weights corresponding to each angle observation. Must be broadcastable to the shape of samples.
    - high: float, optional
        Upper boundary of the principal value of an angle. Default is 2*pi.
    - low: float, optional
        Lower boundary of the principal value of an angle. Default is 0.
    - axis: int or None, optional
        Axis along which means are computed. The default is to compute the mean of the flattened array.
    - nan_policy: {'propagate', 'omit', 'raise'}, optional
        Defines how to handle when input contains nan. Default is 'propagate'.
    - keepdims: bool, optional
        If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

    Returns:
    - circmean: float or ndarray
        Circular mean.
    """
    samples = np.asarray(samples)

    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape != samples.shape:
            try:
                weights = np.broadcast_to(weights, samples.shape)
            except ValueError:
                raise ValueError("weights and samples must be broadcastable to the same shape")
    else:
        weights = np.ones_like(samples)

    # Handle NaNs according to nan_policy
    if nan_policy == 'omit':
        mask = ~np.isnan(samples)
        samples = np.where(mask, samples, 0)
        weights = np.where(mask, weights, 0)
        count = np.sum(mask * weights, axis=axis, keepdims=keepdims)
    elif nan_policy == 'raise':
        if np.isnan(samples).any():
            raise ValueError("Input contains NaNs")
        count = np.sum(weights, axis=axis, keepdims=keepdims)
    elif nan_policy == 'propagate':
        if np.isnan(samples).any():
            return np.full(samples.shape if axis is None else np.delete(samples.shape, axis), np.nan)
        count = np.sum(weights, axis=axis, keepdims=keepdims)
    else:
        raise ValueError("nan_policy must be 'propagate', 'omit', or 'raise'")

    # Normalize samples to [0, 2*pi)
    angle_range = high - low
    angles = (samples - low) * (2 * np.pi / angle_range)

    # Compute weighted sum of unit vectors
    complex_sum = np.sum(weights * np.exp(1j * angles), axis=axis, keepdims=keepdims)
    mean_vector = complex_sum / count

    # Compute mean angle
    mean_angle = np.angle(mean_vector)

    # Map result back to [low, high)
    mean_angle = (mean_angle * (angle_range / (2 * np.pi))) + low
    mean_angle = np.mod(mean_angle - low, angle_range) + low

    return mean_angle





def pairwise_phase_consistecy(I_spk_adjusted,total_simul = 200,total_samples = [1000]):
    # total_samples = [10,100,1000]
    # total_simul = 200
    r_ppc = []
    for simul in range(total_simul):
        ppc_pre = []
        for n_of_data in total_samples:

            samples = np.random.choice(I_spk_adjusted,n_of_data)
            crss = np.exp(1j*(samples))

            dof = np.nansum(~np.isnan(crss))
            sinSum = np.abs(np.nansum(np.imag(crss)))
            cosSum = np.nansum(np.real(crss))
            ppc_pre.append((cosSum**2+sinSum**2 - dof)/(dof*(dof-1)))


        r_ppc.append(ppc_pre)
    r_ppc = np.array(r_ppc)

    return r_ppc


def continuous_firing_rate(spike_times, time_vector, sampling_rate, sigma_points):
    """
    Calculate the continuous firing rate from a spike train.

    Parameters
    ----------
    spike_times : numpy.ndarray
        Array containing spike times.
    time_vector : numpy.ndarray
        Time vector.
    sampling_rate : float
        Time vector sampling rate.
    sigma_points : float
        Standard deviation parameter for the Gaussian kernel.

    Returns
    -------
    spike_vector : numpy.ndarray
        Spike vector based on the provided time vector.
    spike_vector_smoothed : numpy.ndarray
        Smoothed spike vector representing the firing rate.

    Notes
    -----
    This function calculates the firing rate by convolving a spike train with a Gaussian kernel
    to produce a continuous representation of the firing rate.

    It assumes that the spike times are in the same units as the time vector.
    The sigma_points parameter controls the degree of smoothing applied to the firing rate.
    A typical way to set sigma_points is in relation to time, for instance:
    sigma_points = 0.01 * sampling_rate for smoothing over a gaussian window with 10 millisecond std
     (assuming sampling_rate' in Hz).
    """


    x_values = np.arange(-10.0 * sigma_points, 10.0 * sigma_points + 1.0)
    gaussian_kernel = np.exp(-((x_values**2) / (2 * (sigma_points**2))))
    
    dt = 1 / sampling_rate
    time_vector_bins = time_vector - dt / 2
    time_vector_bins = np.append(time_vector_bins, time_vector[-1] + dt / 2)
    spike_vector, _ = np.histogram(spike_times, bins=time_vector_bins)
    
    spike_vector_smoothed = sig.convolve(spike_vector, gaussian_kernel, mode='same')
    spike_vector_smoothed = spike_vector_smoothed/np.nansum(gaussian_kernel)
    return spike_vector, spike_vector_smoothed



'''
def timestamps_to_idx(timestamps,timevector):
    # timestamps in the same time unit as in timevector
    # returns indexes such that timevector[idx_timestamps] = timestamps
    # with an error less than 100 ms.
    
    if len(timestamps)>0:
        idx_timestamps = searchsorted2(timevector, timestamps)
        I_keep = np.abs((timevector[idx_timestamps]-timestamps))<0.1
        idx_timestamps = idx_timestamps[I_keep]
    else:
        idx_timestamps = []
    return idx_timestamps

'''

def bin_spike_counts(spike_times, time_vector, bin_duration):
    """
    Bins spike counts within specified time intervals.

    Parameters:
    - spike_times (array-like): Array containing spike times (same unit as in time_vector)
    - time_vector (array-like): Array containing the time vector.
    - bin_duration (float): Duration of each time bin.

    Returns:
    - spike_vector (array-like): Array containing the binned spike counts.
    - time_vector_center_bins (array-like): Array containing the center time of each bin.
    """
    # Generate bins for the time vector
    time_vector_bins = np.arange(time_vector[0], time_vector[-1] + bin_duration, bin_duration)
    # Calculate the center of each bin
    time_vector_center_bins = time_vector_bins[:-1] + bin_duration / 2

    # Count the number of spikes in each bin
    spike_vector, _ = np.histogram(spike_times, bins=time_vector_bins)

    return spike_vector, time_vector_center_bins


def get_assembly_membership_correlation(activity_matrix,assemblyAct,neuron_labels,threshold=0.5):
    
    assembly_members = []
    for assembl in range(assemblyAct.shape[0]):
        correlation_coefficients = []
        for cell in range(0,activity_matrix.shape[0]):
            valid_mask = ~np.isnan(assemblyAct[assembl,:])
            correlation_coefficients.append(np.corrcoef(activity_matrix[cell,valid_mask],assemblyAct[assembl,valid_mask])[0,1])
        correlation_coefficients = np.array(correlation_coefficients)
        
        memb_idx = np.where(correlation_coefficients > np.nanmean(correlation_coefficients) + threshold*np.nanstd(correlation_coefficients))[0]
        neuron_labels_assembly = decode.select(neuron_labels,memb_idx,concatenate=False)
        assembly_members.append(neuron_labels_assembly)
        
    return assembly_members

def get_assembly_membership(patterns,neuron_labels,threshold=0.5):
    assembly_members = []
    for pattern in patterns:
        argmax_abs_value = np.argmax(np.abs(pattern))
        
        if pattern[argmax_abs_value] < 0:
            pattern *= -1

        memb_idx = np.where(pattern > np.nanmean(pattern) + threshold*np.nanstd(pattern))[0]
        neuron_labels_assembly = decode.select(neuron_labels,memb_idx,concatenate=False)
        assembly_members.append(neuron_labels_assembly)
        
    return assembly_members
    


    from scipy.stats import zscore

def compute_spike_triggered_average(matrix, refbins, window_length):
    """
    Computes the spike-triggered average for given matrix following reference bins.

    Parameters:
    - matrix (ndarray): The activity matrix.
    - refbins (ndarray): Array of indices corresponding to significant activations.
    - window_length (int): The number of bins to include in the window (should be odd for symmetry).

    Returns:
    - ndarray: The spike-triggered average matrix.
    - ndarray: Relative bin positions.
    """
    trigav = np.zeros((len(matrix), window_length))
    relative_bin = np.arange(-int(window_length / 2), int(window_length / 2) + 1)

    if len(refbins) > 0:
        for refbin in refbins:
            bins_ = relative_bin + refbin
            valid_bins = (bins_ >= 0) & (bins_ < matrix.shape[1])
            trigav[:, valid_bins] += matrix[:, bins_[valid_bins]]
        trigav /= len(refbins)
    else:
        print("No active bins found for the selected assembly.")

    return trigav, relative_bin


def circular_sort(phase_values):
    # Convert phase values to complex numbers
    complex_numbers = np.exp(1j * phase_values)
    
    # Compute mean of the complex numbers
    mean_complex = np.mean(complex_numbers)
    
    # Compute angle (phase) of the mean complex number
    mean_phase = np.angle(mean_complex)
    
    # Compute circular distances from the mean phase
    circular_distances = np.angle(complex_numbers * np.exp(-1j * mean_phase))
    
    # Sort phase values based on circular distances
    sorted_indices = np.argsort(circular_distances)
    sorted_phase_values = phase_values[sorted_indices]
    
    return sorted_indices,sorted_phase_values


def integer_count(array, min_val=0, max_val=6):
    # Initialize a dictionary to store counts for each integer value
    counts = {i: 0 for i in range(min_val, max_val + 1)}
    
    # Iterate over the array and update counts
    for num in array:
        if min_val <= num <= max_val:
            counts[num] += 1
    
    return counts



def confidence_interval(data, confidence=0.95, axis=None):
    """
    Calculate the confidence interval for the mean of a dataset along a specified axis.

    Parameters:
    - data (array-like): The dataset for which to calculate the confidence interval.
    - confidence (float): The confidence level for the interval (default is 0.95).
    - axis (int, optional): The axis along which to calculate the mean and confidence interval. 
                            If None, the data is flattened.

    Returns:
    - (ndarray, ndarray): The lower and upper bounds of the confidence interval.
    """
    data = np.array(data)
    n = data.shape[axis] if axis is not None else len(data)
    mean = np.nanmean(data, axis=axis)
    se = stats.sem(data, axis=axis, nan_policy='omit')  # Standard error of the mean

    if n > 30:
        # Use z-score for large samples
        h = se * stats.norm.ppf((1 + confidence) / 2)
    else:
        # Use t-score for small samples
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean - h, mean + h




def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """
    Detect peaks in data based on amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        Data to search for peaks.
    mph : {None, number}, optional (default=None)
        Minimum peak height. Peaks must exceed this value.
    mpd : int, optional (default=1)
        Minimum peak distance. Peaks must be separated by at least this many data points.
    threshold : float, optional (default=0)
        Minimum difference between a peak and its immediate neighbors.
    edge : {'rising', 'falling', 'both', None}, optional (default='rising')
        Type of edge to detect flat peaks: 'rising', 'falling', 'both', or None.
    kpsh : bool, optional (default=False)
        Keep peaks with the same height even if they are closer than `mpd`.
    valley : bool, optional (default=False)
        If True, detect valleys (local minima) instead of peaks.
    show : bool, optional (default=False)
        If True, plot the data and the detected peaks.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes on which to plot if `show` is True.

    Returns
    -------
    ind : 1D array_like
        Indices of the detected peaks in `x`.

    Notes
    -----
    To detect valleys instead of peaks, the input signal is inverted.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(100)
    >>> peaks = detect_peaks(x, show=True)
    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    
    if valley:
        x = -x
    
    dx = np.diff(x)
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf

    # Detect edges
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    if edge in ['rising', 'both']:
        ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
    if edge in ['falling', 'both']:
        ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]

    ind = np.unique(np.hstack((ine, ire, ife)))

    # Exclude indices near NaNs
    if ind.size and indnan.size:
        invalid = np.unique(np.hstack((indnan, indnan - 1, indnan + 1)))
        ind = ind[~np.in1d(ind, invalid)]

    # Filter by minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]

    # Apply threshold
    if ind.size and threshold > 0:
        ind = ind[(x[ind] - np.maximum(x[ind - 1], x[ind + 1])) > threshold]

    # Enforce minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])[::-1]]  # Sort by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                close = (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd)
                if not kpsh:
                    close &= x[ind[i]] > x[ind]
                idel |= close
                idel[i] = False  # Keep current peak
        ind = np.sort(ind[~idel])

    # Optional plotting
    if show:
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(x, label='Data')
        ax.plot(ind, x[ind], 'ro', label='Peaks')
        ax.legend()
        ax.set_title("Detected Peaks")
        plt.show()

    return ind



def get_ripple_info_AmpEnvMethod(lfp_signal, sampling_rate, ripple_threshold_level, 
                                 ripple_low_freq, ripple_high_freq, min_ripple_duration, 
                                 max_ripple_duration, smooth_time):
    """
    Detects ripples in the LFP data using amplitude envelope and thresholding.

    Returns
    -------
    ripple_features : dict
        Dictionary containing all ripple properties:
            - 'centers': Indices of center amplitudes for detected ripples.
            - 'powers': Sum of amplitudes for each detected ripple.
            - 'cycles': Indices of samples for each detected ripple.
            - 'minima': Indices of minima in the raw signal for each detected ripple.
            - 'max_amplitudes': Maximum envelope amplitude for each ripple.
    """
    smooth_points = int(sampling_rate * smooth_time)
    filtered_signal = eegfilt(lfp_signal, sampling_rate, ripple_low_freq, ripple_high_freq)
    envelope = np.abs(hilbert(filtered_signal))
    smoothed_envelope = gaussian_smooth_1d(envelope, smooth_points)
    # smoothed_envelope = np.abs(hf.hilbert(filtered_signal))

    peak_threshold = np.nanmean(smoothed_envelope) + ripple_threshold_level * np.nanstd(smoothed_envelope)
    max_ripple_samples = int(sampling_rate * max_ripple_duration)
    min_ripple_samples = int(sampling_rate * min_ripple_duration)

    peak_indices = detect_peaks(smoothed_envelope, mph=peak_threshold, mpd=(1/15)*sampling_rate, edge='rising')

    ripple_features = {
        "centers": [],
        "powers": [],
        "cycles": [],
        "minima": [],
        "max_amplitudes": [],
    }

    one_sd_threshold = np.nanmean(smoothed_envelope) + np.nanstd(smoothed_envelope)

    for peak_idx in peak_indices:
        start_idx = peak_idx
        while start_idx > 0 and smoothed_envelope[start_idx] > one_sd_threshold:
            start_idx -= 1

        end_idx = peak_idx
        while end_idx < len(smoothed_envelope) - 1 and smoothed_envelope[end_idx] > one_sd_threshold:
            end_idx += 1

        ripple_window = np.arange(start_idx, end_idx + 1)
        ripple_length = len(ripple_window)
        if ripple_length < min_ripple_samples or ripple_length > max_ripple_samples:
            continue

        center_idx = peak_idx
        min_idx = ripple_window[np.argmin(filtered_signal[ripple_window])]

        ripple_features["centers"].append(center_idx)
        ripple_features["powers"].append(np.nansum(envelope[ripple_window]))
        ripple_features["max_amplitudes"].append(np.nanmax(envelope[ripple_window]))
        ripple_features["cycles"].append(ripple_window)
        ripple_features["minima"].append(min_idx)

    return ripple_features
