import numpy as np
from scipy import signal as sig
import math
import os
import re
from scipy import signal
import numpy as np
import scipy.io
import h5py 
import xml.etree.ElementTree as ET

base_path = "/media/apollo/projects/hc-18/data/"
### load LFP data from HC18 data set
def load_lfp_hc18_data(session):
    

    path = base_path + session
    os.chdir(path)

    # get LFP
    LFP = []
    inputfilename = session + '.lfp'

    with open(inputfilename, 'rb') as fid:
        LFP = np.fromfile(fid, np.int16)

    # get parameters
    filepath = session + ".xml"
    tree = ET.parse(filepath)
    root = tree.getroot()
    for child in root.findall('fieldPotentials'):
        srate = float(child.find('lfpSamplingRate').text)
    for child in root.findall('acquisitionSystem'):
        Nchan = int(child.find('nChannels').text)


    # organize LFP    
    LFP = np.reshape(LFP, (int(len(LFP)/Nchan),int(Nchan))).transpose()
    # remember that LFP is int16. It takes less space but we should transform it to float.
    # here we will use int16, but remember to manipulate it with float
    return LFP,srate



def get_electrodes(session):
    
    path = base_path + session
    os.chdir(path)

    # get parameters
    filepath = session + ".xml"
    tree = ET.parse(filepath)
    root = tree.getroot()

    # Find only the <spikeDetection> section
    spike_detection = root.find("spikeDetection")
    
    # If <spikeDetection> exists, extract channels from each group
    channel_groups = []
    if spike_detection is not None:
        for group in spike_detection.findall(".//channelGroups/group"):
            channels = [int(ch.text) for ch in group.find("channels").findall("channel")]
            channel_groups.append(channels)
    
    return channel_groups



def load_evt(session):

    path = base_path + session
    os.chdir(path)
    
    filename = session + '.cat.evt'

    # Initialize events dictionary
    events = {'time': [], 'description': []}
    
    # Check if file exists
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{filename}' not found.")
    except PermissionError:
        raise PermissionError(f"Cannot read {filename} (insufficient access rights?).")
    
    # Parse lines using regular expressions
    for line in lines:
        match = re.match(r'([^ \t]*).*', line)
        if match:
            time = match.group(1)
            events['time'].append(float(time))
            description = re.sub(r'[^ \t]*[ \t]*', '', line, count=1)
            events['description'].append(description.strip())
    
    # Convert time to seconds
    if events['time']:
        events['time'] = [t / 1000 for t in events['time']]
    
    return events

def get_events(session,event_name):
    
    data_path = base_path + session
    events = load_evt(session)
    
    event_times = []
    event_descriptions = []
    for i, description in enumerate(events['description']):
        if event_name in description and ("beginning" in description or "end" in description):
            event_times.append(events['time'][i])
            event_descriptions.append(description)
    return event_times, event_descriptions




def load_pos(session):

    # The sampling rate of the digital video camera was 30Hz. We extracted the positions of LEDs 
    # from the video movie, synchronized the video movie and electrophysiological data (.eeg file), 
    # and generated .whl files by interpolate the LEDs positions at 1/32 of
    # the sampling rate of the .eeg file.
    # In most cases, the sampling rate of .eeg files was 1250Hz, therefore 
    # the sampling rate of the .whl files was 1250/32 = 39.0625 Hz.

    path = base_path + session
    os.chdir(path)
    
    filename = session + '.pos'

    x_coordinates = []
    y_coordinates = []
    with open(filename, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            if len(columns) >= 2:  # Ensure there are at least two columns
                x_coordinates.append(int(columns[0]))
                y_coordinates.append(int(columns[1]))
    x_coordinates = np.array(x_coordinates)
    y_coordinates = np.array(y_coordinates)
    
    # get parameters
    filepath = session + ".xml"
    tree = ET.parse(filepath)
    root = tree.getroot()
    for child in root.findall('video'):
        video_srate = float(child.find('samplingRate').text)

    track_time_vector = np.linspace(0,x_coordinates.shape[0]/video_srate,x_coordinates.shape[0])

    return x_coordinates, y_coordinates, track_time_vector, video_srate


def get_shanks_info(session):

    path = base_path + session
    os.chdir(path)


    reject_channels = dict()
    reject_channels['Train-242-20140124'] = [0,26,56,57,58,59,60,61,62,63]
    reject_channels['Train-261-20140617'] = [0,24,33]
    reject_channels['Train-272-20141215'] = [0]
    reject_channels['Train-292-20150501'] = [10,11,17,23,26,32,51,79,80,82,85,88]
    reject_channels['Train-314-20160118'] = [59,66,79,107]

    good_shanks = dict()
    good_shanks['Train-242-20140124'] = [0,1,2,3,4,5,7,8,9,10,11,13]
    good_shanks['Train-261-20140617'] = [4,5,6,8,9,10,11,12,13,14,15]
    good_shanks['Train-272-20141215'] = [0,3,7,10,11,12,13,14]
    good_shanks['Train-292-20150501'] = [0,3,4,5,6,7,9,12]
    good_shanks['Train-314-20160118'] = [0,1,2,3,4,6,7,10,11,12,13,15]

    # Train-242-20140124
    # shanks = [0,1,2,3,4,5,7,8,9,10,11,13]
    
    # Train-261-20140617
    # shanks = [4,5,6,8,9,10,11,12,13,14,15]
    
    # Train-272-20141215
    # shanks = [0,3,7,10,11,12,13,14]
    
    # Train-292-20150501
    # shanks = [0,3,4,5,6,7,9,12]
    
    # Train-314-20160118
    # shanks = [0,1,2,3,4,6,7,10,11,12,13,15]
    
        
    # I think electrodes (octrodes) from Train-314-20160118 are incorrect, 
    # so we will use the electrodes from Train-292-20150501
    if session == 'Train-314-20160118':
        # Load the XML file
        tree = ET.parse(f'{base_path}/Train-292-20150501/Train-292-20150501.xml')  # Provide the actual filename
    else:
        tree = ET.parse(f'{session}.xml')  # Provide the actual filename

    root = tree.getroot()

    # Initialize the list to store channel groups
    shanks_electrodes = []

    # Find all channel groups
    channel_groups = root.findall(".//channelGroups/group")

    # Extract channels from each group
    for group in channel_groups:
        channels = [int(channel.text) for channel in group.findall("channel")]
        if channels:  # Check if the list of channels is not empty
            shanks_electrodes.append(np.setdiff1d(channels,reject_channels[session]))
    
    left_shanks = np.arange(0,8)
    right_shanks = np.arange(8,16)

        
    for shk in range(16):
        if shk not in good_shanks[session]:
            shanks_electrodes[shk] = []
            
    return shanks_electrodes,left_shanks,right_shanks


def get_shanks(shanks_electrodes,channel_groups):
        
    matched_shanks = []
    for group in channel_groups:
        for i, shank in enumerate(shanks_electrodes):
            if all(ch in shank for ch in group):
                matched_shanks.append(i)
                break
    
    return matched_shanks


    
def split_into_contiguous_blocks(arr):
    result = []
    current_block = []

    for i, val in enumerate(arr):
        if i == 0:
            current_block.append(val)
        else:
            # Check if current value continues the sequence
            if val == arr[i - 1] + 1:
                current_block.append(val)
            else:
                # End of a block
                result.append(current_block)
                current_block = [val]
    
    # Append the last block
    if current_block:
        result.append(current_block)

    return result
