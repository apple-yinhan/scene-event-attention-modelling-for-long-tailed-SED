#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:54:19 2023

@author: bfzystudent
"""
import os
import librosa
import numpy as np
import wave
from dcase_util.containers import metadata


def load_audio(filename, mono=True, fs=44100):
    """Load audio file into numpy array
    Supports 24-bit wav-format

    Taken from TUT-SED system: https://github.com/TUT-ARG/DCASE2016-baseline-system-python

    Parameters
    ----------
    filename:  str
        Path to audio file

    mono : bool
        In case of multi-channel audio, channels are averaged into single channel.
        (Default value=True)

    fs : int > 0 [scalar]
        Target sample rate, if input audio does not fulfil this, audio is resampled.
        (Default value=44100)

    Returns
    -------
    audio_data : numpy.ndarray [shape=(signal_length, channel)]
        Audio

    sample_rate : integer
        Sample rate

    """

    file_base, file_extension = os.path.splitext(filename)
    if file_extension == '.wav':
        _audio_file = wave.open(filename)

        # Audio info
        sample_rate = _audio_file.getframerate()
        sample_width = _audio_file.getsampwidth()
        number_of_channels = _audio_file.getnchannels()
        number_of_frames = _audio_file.getnframes()

        # Read raw bytes
        data = _audio_file.readframes(number_of_frames)
        _audio_file.close()

        # Convert bytes based on sample_width
        num_samples, remainder = divmod(len(data), sample_width * number_of_channels)
        if remainder > 0:
            raise ValueError('The length of data is not a multiple of sample size * number of channels.')
        if sample_width > 4:
            raise ValueError('Sample size cannot be bigger than 4 bytes.')

        if sample_width == 3:
            # 24 bit audio
            a = np.empty((num_samples, number_of_channels, 4), dtype=np.uint8)
            raw_bytes = np.fromstring(data, dtype=np.uint8)
            a[:, :, :sample_width] = raw_bytes.reshape(-1, number_of_channels, sample_width)
            a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
            audio_data = a.view('<i4').reshape(a.shape[:-1]).T
        else:
            # 8 bit samples are stored as unsigned ints; others as signed ints.
            dt_char = 'u' if sample_width == 1 else 'i'
            a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
            audio_data = a.reshape(-1, number_of_channels).T

        if mono:
            # Down-mix audio
            audio_data = np.mean(audio_data, axis=0)

        # Convert int values into float
        audio_data = audio_data / float(2 ** (sample_width * 8 - 1) + 1)

        # Resample
        if fs != sample_rate:
            audio_data = librosa.core.resample(audio_data, sample_rate, fs)
            sample_rate = fs

        return audio_data, sample_rate
    return None, None

def create_folder(_fold_path):
    if not os.path.exists(_fold_path):
        os.makedirs(_fold_path)
        
def extract_mbe(_y, _sr, _nfft, _hop, _nb_mel, _fmin, _fmax):
    spec, _ = librosa.core.spectrum._spectrogram(y=_y, n_fft=_nfft, hop_length=_hop, power=1)
    mel_basis = librosa.filters.mel(sr=_sr, n_fft=_nfft, n_mels=_nb_mel, fmin=_fmin, fmax=_fmax)
    mbe = np.dot(mel_basis, spec)
    mbe = librosa.amplitude_to_db(mbe, ref=np.max)*(-1)
    return mbe

def load_labels(file_name, nframes, class_labels, sampling_rate, hop_size):
    annotations = []
    for l in open(file_name):
        words = l.strip().split('\t')
        
        annotations.append([float(words[0]), float(words[1]), 
                                class_labels[words[2]], float(words[3])])
    # Initialize label matrix
    label = np.zeros((nframes, len(class_labels)))
    tmp_data = np.array(annotations)
    
    frame_start = np.floor(tmp_data[:, 0] * sampling_rate / hop_size).astype(int)
    frame_end = np.ceil(tmp_data[:, 1] * sampling_rate / hop_size).astype(int)
    se_class = tmp_data[:, 2].astype(int)
    for ind, val in enumerate(se_class):
        label[frame_start[ind]:frame_end[ind], val] = tmp_data[:, 3][ind]

    return label

def find_contiguous_regions(activity_array):
    # Find the changes in the activity_array
    change_indices = np.logical_xor(activity_array[1:], activity_array[:-1]).nonzero()[0]
    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = np.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = np.r_[change_indices, len(activity_array)]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))

def process_event(class_labels, frame_probabilities, threshold, hop_length_seconds,
                 ):
    results = []
    for event_id, event_label in enumerate(class_labels):
        # Make sure that the evaluated labels are the ones that correspond to the hard labels
        if event_label in class_labels:
            # Binarization
            event_activity = frame_probabilities[event_id, :] > threshold

            # Convert active frames into segments and translate frame indices into time stamps
            event_segments = np.floor(find_contiguous_regions(event_activity) * hop_length_seconds)

            # Store events
            for event in event_segments:
                results.append(
                    metadata.MetaDataItem(
                        {
                            'event_onset': event[0],
                            'event_offset': event[1],
                            'event_label': event_label
                        }
                    )
                )
    
    results = metadata.MetaDataContainer(results)

    # Event list post-processing
    results = results.process_events(minimum_event_length=None, minimum_event_gap=0.1)#0.1
    results = results.process_events(minimum_event_length=0.1, minimum_event_gap=None)
    return results