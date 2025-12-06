import os
import numpy as np
import scipy.io

import helper_functions as hf
import normalizing_functions as nf
import smoothing_functions as smooth


def get_ripple_info_from_envelope(envelope, sampling_rate, ripple_threshold_level = 2, 
                    min_ripple_duration = 0.02, max_ripple_duration = 0.2, smooth_time = 0.005):
    
    """
    Detects ripples in the LFP data using amplitude envelope and thresholding.

    Parameters
    ----------
    lfp_signal : array_like
        The LFP signal to analyze.
    sampling_rate : float
        Sampling rate of the signal (Hz).
    ripple_threshold_level : float
        Threshold for ripple detection, in units of the standard deviation.
    ripple_low_freq : float
        Lower frequency bound for ripple bandpass filter (Hz).
    ripple_high_freq : float
        Upper frequency bound for ripple bandpass filter (Hz).
    min_ripple_duration : float
        Minimum ripple duration (seconds).
    max_ripple_duration : float
        Maximum ripple duration (seconds).
    smooth_time : float
        Smoothing window (seconds) used in the ripple amplitude envelope.

    Returns
    -------
    ripple_features : dict
        Dictionary containing all ripple properties:
            - 'centers': Indices of center amplitudes for detected ripples.
            - 'powers': Sum of amplitudes for each detected ripple.
            - 'cycles': Indices of samples for each detected ripple.
            - 'max_amplitudes': Maximum envelope amplitude for each ripple.
    """
    

    sigma_points = smooth.get_sigma_points(sigma = smooth_time,sampling_rate = sampling_rate)
    kernel, x_values = smooth.generate_1d_gaussian_kernel(sigma_points)
    smoothed_envelope = smooth.gaussian_smooth_1d(envelope, kernel)


    peak_threshold = np.nanmean(smoothed_envelope) + ripple_threshold_level * np.nanstd(smoothed_envelope)
    max_ripple_samples = int(sampling_rate * max_ripple_duration)
    min_ripple_samples = int(sampling_rate * min_ripple_duration)

    peak_indices = hf.detect_peaks(smoothed_envelope, mph=peak_threshold, mpd=max_ripple_samples, edge='rising')

    ripple_features = {
        "centers": [],
        "powers": [],
        "cycles": [],
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

        ripple_features["centers"].append(center_idx)
        ripple_features["powers"].append(np.nansum(envelope[ripple_window]))
        ripple_features["max_amplitudes"].append(np.nanmax(envelope[ripple_window]))
        ripple_features["cycles"].append(ripple_window)

    return ripple_features

    
def get_ripple_info(lfp_signal, sampling_rate, ripple_band=(100, 250),ripple_threshold_level = 2, 
                                 min_ripple_duration = 0.02, max_ripple_duration = 0.2, smooth_time = 0.005):
    
    """
    Detects ripples in the LFP data using amplitude envelope and thresholding.

    Parameters
    ----------
    lfp_signal : array_like
        The LFP signal to analyze.
    sampling_rate : float
        Sampling rate of the signal (Hz).
    ripple_threshold_level : float
        Threshold for ripple detection, in units of the standard deviation.
    ripple_low_freq : float
        Lower frequency bound for ripple bandpass filter (Hz).
    ripple_high_freq : float
        Upper frequency bound for ripple bandpass filter (Hz).
    min_ripple_duration : float
        Minimum ripple duration (seconds).
    max_ripple_duration : float
        Maximum ripple duration (seconds).
    smooth_time : float
        Smoothing window (seconds) used in the ripple amplitude envelope.

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
    
    filtered_signal = hf.eegfilt(lfp_signal, sampling_rate, ripple_band[0], ripple_band[1])
    envelope = np.abs(hf.hilbert(filtered_signal))

    sigma_points = smooth.get_sigma_points(sigma = smooth_time,sampling_rate = sampling_rate)
    kernel, x_values = smooth.generate_1d_gaussian_kernel(sigma_points)
    smoothed_envelope = smooth.gaussian_smooth_1d(envelope, kernel)


    peak_threshold = np.nanmean(smoothed_envelope) + ripple_threshold_level * np.nanstd(smoothed_envelope)
    max_ripple_samples = int(sampling_rate * max_ripple_duration)
    min_ripple_samples = int(sampling_rate * min_ripple_duration)

    peak_indices = hf.detect_peaks(smoothed_envelope, mph=peak_threshold, mpd=max_ripple_samples, edge='rising')

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

def compute_theta_delta_ratio(lfp_signal, sampling_rate, theta_band=(4, 12), delta_band=(0, 4), window_duration=2):
    """
    Computes the theta/delta power ratio for an LFP signal using a sliding window with convolution.

    Parameters
    ----------
    lfp_signal : array_like
        The LFP signal to analyze.
    sampling_rate : float
        Sampling rate of the signal (Hz).
    theta_band : tuple, optional
        Frequency range for theta (default is 4-12 Hz).
    delta_band : tuple, optional
        Frequency range for delta (default is 1-4 Hz).
    window_duration : float, optional
        Duration (seconds) of the sliding window for ratio calculation (default is 2 seconds).

    Returns
    -------
    theta_delta_ratios : array_like
        Theta/Delta power ratios for the entire signal.
    """
    # Calculate window size in samples
    window_samples = int(window_duration * sampling_rate)

    # Bandpass filter to extract theta and delta components
    theta_signal = hf.eegfilt(lfp_signal, sampling_rate, theta_band[0], theta_band[1])
    delta_signal = hf.eegfilt(lfp_signal, sampling_rate, delta_band[0], delta_band[1])

    theta_power = np.abs(hf.hilbert(theta_signal))**2
    delta_power = np.abs(hf.hilbert(delta_signal))**2

    theta_power = nf.min_max_norm(theta_power)
    delta_power = nf.min_max_norm(delta_power)
    
    # Define a normalized sliding window (boxcar) kernel
    kernel = np.ones(window_samples) / window_samples

    # Use convolution to calculate sliding window mean for theta and delta power
    theta_mean = np.convolve(theta_power, kernel, mode='same')
    delta_mean = np.convolve(delta_power, kernel, mode='same')

    # Compute the ratio directly
    theta_delta_ratios = theta_mean / (delta_mean + 1e-10)  # Avoid division by zero

    return theta_delta_ratios

def validate_ripples_theta_delta(theta_delta_ratios, ripple_features,ratio_threshold):
    """
    Validates and filters detected ripples based on the theta/delta power ratio using ripple cycles.

    Parameters
    ----------
    theta_delta_ratios : array_like
        Precomputed theta/delta power ratios for the entire signal.
    ripple_features : dict
        Dictionary containing all ripple properties:
            - 'centers': Indices of center amplitudes for detected ripples.
            - 'powers': Sum of amplitudes for each detected ripple.
            - 'cycles': Indices of samples for each detected ripple.
            - 'minima': Indices of minima in the raw signal for each detected ripple.
            - 'max_amplitudes': Maximum envelope amplitude for each ripple.

    Returns
    -------
    filtered_features : dict
        Ripple features filtered based on delta dominance.
    valid_ripples : list
        Indices of valid ripple cycles.
    """
    # Initialize filtered features
    filtered_features = {key: [] for key in ripple_features.keys()}
    valid_ripples = []

    # Validate and filter ripples based on their cycles
    for idx, cycle in enumerate(ripple_features['cycles']):
        # Compute the mean theta/delta ratio for the entire ripple cycle
        mean_ratio = np.nanmean(theta_delta_ratios[cycle])
        if mean_ratio < ratio_threshold:  # Delta dominant
            valid_ripples.append(cycle)
            # Add ripple's features to the filtered dictionary
            for key in ripple_features:
                filtered_features[key].append(ripple_features[key][idx])

    return filtered_features, valid_ripples




def filter_ripples_by_artifact(distant_channel, sampling_rate, ripple_features, 
                               high_frequency_band = (100,250), power_threshold_factor = 2):
    """
    Filters ripples by checking for high-frequency artifacts in a distant channel.

    Parameters
    ----------
    distant_channel : array_like
        The distant channel signal to analyze for artifacts.
    sampling_rate : float
        Sampling rate of the signal (Hz).
    ripple_features : dict
        Ripple features from the detection function.
    ripple_low_freq : float
        Lower frequency bound for ripple bandpass filter (Hz).
    ripple_high_freq : float
        Upper frequency bound for ripple bandpass filter (Hz).
    power_threshold_factor : float
        Threshold factor for artifact detection, in units of the standard deviation of power.

    Returns
    -------
    filtered_features : dict
        Ripple features filtered to exclude those with high-frequency artifacts in the distant channel.
    artifact_flags : list
        Boolean list indicating whether each ripple was excluded due to artifacts.
    """
    # Bandpass filter the distant channel
    filtered_distant_signal = hf.eegfilt(distant_channel, sampling_rate, high_frequency_band[0], high_frequency_band[1])
    envelope_distant_signal = np.abs(hf.hilbert(filtered_distant_signal))
    
    # Compute the artifact detection threshold
    power_threshold = np.nanmean(envelope_distant_signal) + power_threshold_factor * np.nanstd(envelope_distant_signal)

    # Initialize filtered features and artifact flags
    filtered_features = {key: [] for key in ripple_features.keys()}
    artifact_flags = []

    # Iterate through each ripple
    for ripple_idx, ripple_window in enumerate(ripple_features["cycles"]):
        # Check mean power in the ripple window
        window_power = np.nanmean(envelope_distant_signal[ripple_window])
        is_artifact = window_power > power_threshold

        # Record artifact flag
        artifact_flags.append(is_artifact)

        # Keep only ripples without artifacts
        if not is_artifact:
            for key in ripple_features:
                filtered_features[key].append(ripple_features[key][ripple_idx])

    return filtered_features, artifact_flags
