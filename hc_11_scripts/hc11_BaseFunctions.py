import numpy as np
from scipy import signal as sig
import math
import os
import numpy as np
import scipy.io
import h5py 
import xml.etree.ElementTree as ET


### load LFP data from HC11 data set
def load_lfp_hc11_data(RatName):

    path = "/media/apollo/projects/hc-11/data/" + RatName
    os.chdir(path)

    # get LFP
    LFP = []
    inputfilename = RatName + '.eeg'
    
    with open(inputfilename, 'rb') as fid:
        LFP = np.fromfile(fid, np.int16)
    
    # get parameters
    filepath = RatName + ".xml"
    tree = ET.parse(filepath)
    root = tree.getroot()
    for child in root.findall('fieldPotentials'):
        srate = float(child.find('lfpSamplingRate').text)
    for child in root.findall('acquisitionSystem'):
        Nchan = int(child.find('nChannels').text)
    
    if RatName == 'Gatsby_08022013':
        Nchan = 134
    
    # organize LFP    
    LFP = np.reshape(LFP, (int(len(LFP)/Nchan),int(Nchan))).transpose()
    
    # get behavioral epochs
    filepath = RatName + "_sessInfo.mat"
    arrays = {}
    f = h5py.File(filepath)
    
    REM = np.array(f['sessInfo']['Epochs']['REM']).T
    REM_lfp_index = []
    for ii in range(REM.shape[0]):
        REM_lfp_index = np.hstack((REM_lfp_index,np.arange(REM[ii,0]*srate,REM[ii,1]*srate,1)))
    REM_lfp_index = np.unique(REM_lfp_index).astype(int)
    
    Wake = np.array(f['sessInfo']['Epochs']['Wake']).T
    Wake_lfp_index = []
    for ii in range(Wake.shape[0]):
        Wake_lfp_index = np.hstack((Wake_lfp_index,np.arange(Wake[ii,0]*srate,Wake[ii,1]*srate,1)))
    Wake_lfp_index = np.unique(Wake_lfp_index).astype(int)
    
    SWS = np.array(f['sessInfo']['Epochs']['NREM']).T
    SWS_lfp_index = []
    for ii in range(SWS.shape[0]):
        SWS_lfp_index = np.hstack((SWS_lfp_index,np.arange(SWS[ii,0]*srate,SWS[ii,1]*srate,1)))
    SWS_lfp_index = np.unique(SWS_lfp_index).astype(int)
    
    PREEpoch = np.array(f['sessInfo']['Epochs']['PREEpoch']).T
    PREEpoch_lfp_index = []
    for ii in range(PREEpoch.shape[0]):
        PREEpoch_lfp_index = np.hstack((PREEpoch_lfp_index,np.arange(PREEpoch[ii,0]*srate,PREEpoch[ii,1]*srate,1)))
    PREEpoch_lfp_index = np.unique(PREEpoch_lfp_index).astype(int)
    
    MazeEpoch = np.array(f['sessInfo']['Epochs']['MazeEpoch']).T
    MazeEpoch_lfp_index = []
    for ii in range(MazeEpoch.shape[0]):
        MazeEpoch_lfp_index = np.hstack((MazeEpoch_lfp_index,np.arange(MazeEpoch[ii,0]*srate,MazeEpoch[ii,1]*srate,1)))
    MazeEpoch_lfp_index = np.unique(MazeEpoch_lfp_index).astype(int)
    
    POSTEpoch = np.array(f['sessInfo']['Epochs']['POSTEpoch']).T
    POSTEpoch_lfp_index = []
    for ii in range(POSTEpoch.shape[0]):
        POSTEpoch_lfp_index = np.hstack((POSTEpoch_lfp_index,np.arange(POSTEpoch[ii,0]*srate,POSTEpoch[ii,1]*srate+1,1)))
    POSTEpoch_lfp_index = np.unique(POSTEpoch_lfp_index).astype(int)
    
    SWS_lfp_index_pre = np.intersect1d(SWS_lfp_index,PREEpoch_lfp_index)    
    SWS_lfp_index_pos = np.intersect1d(SWS_lfp_index,POSTEpoch_lfp_index)

    return LFP,srate,SWS,SWS_lfp_index,SWS_lfp_index_pre,SWS_lfp_index_pos

    
def load_position_hc11_data(RatName):
    
    import os
    import numpy as np
    import h5py 
  
    path = "/home/rscheffer/dados/hc-11/data/" + RatName
    os.chdir(path)

    # get behavioral epochs
    filepath = RatName + "_sessInfo.mat"
    arrays = {}
    f = h5py.File(filepath)

    two_dim_position = np.squeeze(f['sessInfo']['Position']['TwoDLocation'])
    one_dim_position = np.squeeze(f['sessInfo']['Position']['OneDLocation'])
    behavioral_timevector = np.squeeze(f['sessInfo']['Position']['TimeStamps'])

    PREEpoch_sec = np.squeeze(f['sessInfo']['Epochs']['PREEpoch'])
    MazeEpoch_sec = np.squeeze(f['sessInfo']['Epochs']['MazeEpoch'])
    POSTEpoch_sec = np.squeeze(f['sessInfo']['Epochs']['POSTEpoch'])

    return two_dim_position,one_dim_position,behavioral_timevector,PREEpoch_sec,MazeEpoch_sec,POSTEpoch_sec



def load_spikes_hc11_data(RatName):
    
    import os
    import numpy as np
    import h5py 
    
    path = "/home/rscheffer/dados/hc-11/data/" + RatName
    os.chdir(path)

    # get behavioral epochs
    filepath = RatName + "_sessInfo.mat"
    arrays = {}
    f = h5py.File(filepath)
    
    spike_times = np.squeeze(f['sessInfo']['Spikes']['SpikeTimes'])
    spike_ids = np.squeeze(f['sessInfo']['Spikes']['SpikeIDs'])
    pyr_ids = np.squeeze(f['sessInfo']['Spikes']['PyrIDs'])
    int_ids = np.squeeze(f['sessInfo']['Spikes']['IntIDs'])

    return spike_times,spike_ids,pyr_ids,int_ids




def get_shanks_info(RatName):
    Allshanks = []
    if RatName[0:3] == 'Ach' or RatName[0:3] == 'Cic':

        # Achilles and Cicero
        left_shanks = np.arange(0,6)
        shank1 = np.array(np.arange(0,10))
        shank2 = np.array(np.arange(10,20))
        shank3 = np.array(np.arange(20,30))
        shank4 = np.array(np.arange(30,40))
        shank5 = np.array(np.arange(40,50))
        shank6 = np.array(np.arange(50,60))
        
        right_shanks = np.arange(6,12)
        shank7 = np.array(np.arange(64,74))
        shank8 = np.array(np.arange(74,84))
        shank9 = np.array(np.arange(84,94))
        shank10 = np.array(np.arange(94,104))
        shank11 = np.array(np.arange(104,114))
        shank12 = np.array(np.arange(114,124))


        for sk in range(1,13):
            Allshanks.append(eval("shank" + str(sk)))

    else:

        # Buddy and Gatsby
        # left
        left_shanks = np.arange(0,8)
        shank1 = np.array(np.arange(0,8))
        shank2 = np.array(np.arange(8,16))
        shank3 = np.array(np.arange(16,24))
        shank4 = np.array(np.arange(24,32))
        shank5 = np.array(np.arange(32,40))
        shank6 = np.array(np.arange(40,48))
        shank7 = np.array(np.arange(48,56))
        shank8 = np.array(np.arange(56,64))

        # right
        right_shanks = np.arange(8,16)
        shank9  = np.array(np.arange(64,72))
        shank10 = np.array(np.arange(72,80))
        shank11 = np.array(np.arange(80,88))
        shank12 = np.array(np.arange(88,96))
        shank13 = np.array(np.arange(96,104))
        shank14 = np.array(np.arange(104,112))
        shank15 = np.array(np.arange(112,120))
        shank16 = np.array(np.arange(120,128))


        for sk in range(1,17):
            Allshanks.append(eval("shank" + str(sk)))
            
    return Allshanks,left_shanks,right_shanks

def get_spike_groups(RatName):
    # here neuron_left_shanks and neuron_right_shanks just indicate if neuron ID (e.g., 1301)
    # correspond to a right or left shank.
    
    # To find in which shank (left_shanks or right_shanks logic) the neuron was detected, 
    # check get_neuron_shank_map function.
   
    if RatName[0:3] == 'Ach' or RatName[0:3] == 'Cic':
        spike_groups = np.array([1,2,3,4,5,6,8,9,10,11,12,13])
    
    if RatName[0:3] == 'Bud' or RatName[0:3] == 'Gat':
        spike_groups = np.arange(1,16+1)
    
    return spike_groups


def get_neuron_shank_map(RatName):
    
    # This maps an neuron ID to its electrode shank group.
    # For instance, a neuron ID 1301 in achilles correspond to
    # the shank index 11 in the right side. 
    
    shanks_map = dict()
    
    shanks_map['left_shanks'] = []
    shanks_map['neuron_left_spike_groups'] = []
    
    shanks_map['right_shanks'] = []
    shanks_map['neuron_right_spike_groups'] = []
    
    if RatName[0:3] == 'Ach' or RatName[0:3] == 'Cic':
        shanks_map['left_shanks'] = [0,1,2,3,4,5]
        shanks_map['neuron_left_spike_groups'] = [1,2,3,4,5,6]
        
        shanks_map['right_shanks'] = [6,7,8,9,10,11]
        shanks_map['neuron_right_spike_groups'] = [8,9,10,11,12,13]
        

    if RatName[0:3] == 'Bud' or RatName[0:3] == 'Gat':
        shanks_map['left_shanks'] = [0,1,2,3,4,5,6,7]
        shanks_map['neuron_left_spike_groups'] = [1,2,3,4,5,6,7,8]
        
        shanks_map['right_shanks'] = [8,9,10,11,12,13,14,15]
        shanks_map['neuron_right_spike_groups'] = [9,10,11,12,13,14,15]
    
    return shanks_map

    


def eegfilt(LFP, fs, lowcut, highcut, order=3, notch_freq=None, notch_bandwidth=1):
    """
    Apply a bandpass, lowpass, highpass, or notch filter to the input signal (LFP).
    
        Parameters
        ----------
        LFP (array-like): Input signal.
        fs (float): Sampling frequency of the input signal.
        lowcut (float): Low cutoff frequency for the filter.
        highcut (float): High cutoff frequency for the filter.
        order (int, optional): Order of the Butterworth filter (default is 3).
        notch_freq (float, optional): Frequency to be notched out (default is None).
        notch_bandwidth (float, optional): Bandwidth around the notch frequency (default is 1 Hz).

        Returns
        ---------
        filtered (array-like): Filtered signal.
    """
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Apply notch filter if specified
    if notch_freq is not None:
        notch = notch_freq / nyq
        bandwidth = notch_bandwidth / nyq
        b_notch, a_notch = sig.iirnotch(notch, bandwidth)
        LFP = sig.filtfilt(b_notch, a_notch, LFP)
    
    # Apply the bandpass, lowpass, or highpass filter
    if low == 0:
        b, a = sig.butter(order, high, btype='low')
        filtered = sig.filtfilt(b, a, LFP)
    elif high == 0:
        b, a = sig.butter(order, low, btype='high')
        filtered = sig.filtfilt(b, a, LFP) 
    else:
        b, a = sig.butter(order, [low, high], btype='band')
        filtered = sig.filtfilt(b, a, LFP)

    return filtered


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
        hilbert_channel = sig.hilbert(channel_data, next_power_of_2(len(channel_data)))  # Apply Hilbert transform
        hilbert_result[channel_idx] = hilbert_channel[range(len(channel_data))]  # Store the result for each channel

    return hilbert_result.squeeze() if n_channels == 1 else hilbert_result



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




def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False):
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

 

    return ind

