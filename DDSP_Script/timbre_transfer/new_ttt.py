import copy
import os
import time

import crepe
import ddsp

import gin
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tensorflow.compat.v2 as tf

from utils import (squeeze, compute_audio_features,
                   detect_notes, fit_quantile_transform, tf_float32,
                   stft, stft_np, amplitude_to_db, compute_loudness,
                   compute_f0, reset_crepe, pad_or_trim_to_expected_length)

from IPython.display import Audio
from IPython.core.display import display

from tensorflow.python.ops.numpy_ops import np_config
from typing import Any, Dict, Optional, Sequence, Text, TypeVar
from scipy import stats

# /////////////////////////////////////////////////////////////////////

RATE = 16000
TF_OR_NP = 'tensorflow'


Number = TypeVar('Number', int, float, np.ndarray, tf.Tensor)

# DDSP Helper Functions (taken from various modules)

if TF_OR_NP == 'tensorflow':
    TF_OR_NP = True
else:
    TF_OR_NP = False

# from metrics.py


def squeeze(input_vector):
    """Ensure vector only has one axis of dimensionality."""
    if input_vector.ndim > 1:
        return np.squeeze(input_vector)
    else:
        return input_vector


def compute_audio_features(audio,
                           n_fft=2048,
                           sample_rate=RATE,
                           frame_rate=20):
    """Compute features from audio."""
    audio_feats = {'audio': audio}
    audio = squeeze(audio)

    audio_feats['loudness_db'] = compute_loudness(
        audio, sample_rate, frame_rate, n_fft)

    audio_feats['f0_hz'], audio_feats['f0_confidence'] = (
        compute_f0(audio, sample_rate, frame_rate))

    return audio_feats


def tf_float32(x):
    """Ensure array/tensor is a float32 tf.Tensor."""
    if isinstance(x, tf.Tensor):
        return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
    else:
        return tf.convert_to_tensor(x, tf.float32)

# from spectral_ops.py


LD_RANGE = 120.0  # dB


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Differentiable stft in tensorflow, computed in batch."""
    assert frame_size * overlap % 2.0 == 0.0

    # Remove channel dim if present.
    audio = tf_float32(audio)
    if len(audio.shape) == 3:
        audio = tf.squeeze(audio, axis=-1)

    s = tf.signal.stft(
        signals=audio,
        frame_length=int(frame_size),
        frame_step=int(frame_size * (1.0 - overlap)),
        fft_length=int(frame_size),
        pad_end=pad_end)
    return s


def stft_np(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Non-differentiable stft using librosa, one example at a time."""
    assert frame_size * overlap % 2.0 == 0.0
    hop_size = int(frame_size * (1.0 - overlap))
    is_2d = (len(audio.shape) == 2)

    if pad_end:
        n_samples_initial = int(audio.shape[-1])
        n_frames = int(np.ceil(n_samples_initial / hop_size))
        n_samples_final = (n_frames - 1) * hop_size + frame_size
        pad = n_samples_final - n_samples_initial
        padding = ((0, 0), (0, pad)) if is_2d else ((0, pad),)
        audio = np.pad(audio, padding, 'constant')

    def stft_fn(y):
        return librosa.stft(y=y,
                            n_fft=int(frame_size),
                            hop_length=hop_size,
                            center=False).T

    s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
    return s


def amplitude_to_db(amplitude, use_tf=TF_OR_NP):
    """Converts amplitude to decibels."""
    lib = tf if use_tf else np
    log10 = (lambda x: tf.math.log(x) /
             tf.math.log(10.0)) if use_tf else np.log10
    amin = 1e-20  # Avoid log(0) instabilities.
    db = log10(lib.maximum(amin, amplitude))
    db *= 20.0
    return db


def compute_loudness(audio,
                     sample_rate=RATE,
                     frame_rate=20,
                     n_fft=2048,
                     range_db=LD_RANGE,
                     ref_db=20.7,
                     use_tf=TF_OR_NP,
                     pad_end=True):
    """Perceptual loudness in dB, relative to white noise, amplitude=1.
    Function is differentiable if use_tf=True.
    Args:
        audio: Numpy ndarray or tensor. Shape [batch_size, audio_length] or
            [batch_size,].
        sample_rate: Audio sample rate in Hz.
        frame_rate: Rate of loudness frames in Hz.
        n_fft: Fft window size.
        range_db: Sets the dynamic range of loudness in decibles. The minimum
            loudness (per a frequency bin) corresponds to -range_db.
        ref_db: Sets the reference maximum perceptual loudness as given by
            (A_weighting + 10 * log10(abs(stft(audio))**2.0). The default value
            corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a
            slight dependence on fft_size due to different granularity of perceptual
            weighting.
        use_tf: Make function differentiable by using tensorflow.
        pad_end: Add zero padding at end of audio (like `same` convolution).
    Returns:
        Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
    """
    if sample_rate % frame_rate != 0:
        raise ValueError(
            'frame_rate: {} must evenly divide sample_rate: {}.'
            'For default frame_rate: 250Hz, suggested sample_rate: 16kHz or 48kHz'
            .format(frame_rate, sample_rate))

    # Pick tensorflow or numpy.
    lib = tf if use_tf else np

    # Make inputs tensors for tensorflow.
    audio = tf_float32(audio) if use_tf else audio

    # Temporarily a batch dimension for single examples.
    is_1d = (len(audio.shape) == 1)
    audio = audio[lib.newaxis, :] if is_1d else audio

    # Take STFT.
    hop_size = sample_rate // frame_rate
    overlap = 1 - hop_size / n_fft
    stft_fn = stft if use_tf else stft_np
    s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=pad_end)

    # Compute power.
    amplitude = lib.abs(s)
    power_db = amplitude_to_db(amplitude, use_tf=use_tf)

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= ref_db
    loudness = lib.maximum(loudness, -range_db)
    mean = tf.reduce_mean if use_tf else np.mean

    # Average over frequency bins.
    loudness = mean(loudness, axis=-1)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    # Compute expected length of loudness vector
    n_secs = audio.shape[-1] / float(
        sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)

    # Pad with `-range_db` noise floor or trim vector
    loudness = pad_or_trim_to_expected_length(
        loudness, expected_len, -range_db, use_tf=use_tf)
    return loudness


def compute_f0(audio, sample_rate=RATE, frame_rate=20, viterbi=True):
    """Fundamental frequency (f0) estimate using CREPE.
    This function is non-differentiable and takes input as a numpy array.
    Args:
        audio: Numpy ndarray of single audio example. Shape [audio_length,].
        sample_rate: Sample rate in Hz.
        frame_rate: Rate of f0 frames in Hz.
        viterbi: Use Viterbi decoding to estimate f0.
    Returns:
        f0_hz: Fundamental frequency in Hz. Shape [n_frames,].
        f0_confidence: Confidence in Hz estimate (scaled [0, 1]). Shape [n_frames,].
    """

    n_secs = len(audio) / float(sample_rate)  # `n_secs` can have milliseconds
    crepe_step_size = 1000 / frame_rate  # milliseconds
    expected_len = int(n_secs * frame_rate)
    audio = np.asarray(audio)

    # Compute f0 with crepe.
    _, f0_hz, f0_confidence, _ = crepe.predict(
        audio,
        sr=sample_rate,
        model_capacity='tiny',
        viterbi=viterbi,
        step_size=crepe_step_size,
        center=False,
        verbose=0)

    # Postprocessing on f0_hz
    f0_hz = pad_or_trim_to_expected_length(
        f0_hz, expected_len, 0)  # pad with 0
    f0_hz = f0_hz.astype(np.float32)

    # Postprocessing on f0_confidence
    f0_confidence = pad_or_trim_to_expected_length(
        f0_confidence, expected_len, 1)
    # Set nans to 0 in confidence
    f0_confidence = np.nan_to_num(f0_confidence)
    f0_confidence = f0_confidence.astype(np.float32)
    return f0_hz, f0_confidence


def reset_crepe():
    """Reset the global state of CREPE to force model re-building."""
    for k in crepe.core.models:
        crepe.core.models[k] = None


def pad_or_trim_to_expected_length(vector,
                                   expected_len,
                                   pad_value=0,
                                   len_tolerance=20,
                                   use_tf=TF_OR_NP):
    """Make vector equal to the expected length.
    Feature extraction functions like `compute_loudness()` or `compute_f0` produce
    feature vectors that vary in length depending on factors such as `sample_rate`
    or `hop_size`. This function corrects vectors to the expected length, warning
    the user if the difference between the vector and expected length was
    unusually high to begin with.
    Args:
        vector: Numpy 1D ndarray. Shape [vector_length,]
        expected_len: Expected length of vector.
        pad_value: Value to pad at end of vector.
        len_tolerance: Tolerance of difference between original and desired vector
          length.
        use_tf: Make function differentiable by using tensorflow.
    Returns:
        vector: Vector with corrected length.
    Raises:
        ValueError: if `len(vector)` is different from `expected_len` beyond
        `len_tolerance` to begin with.
    """
    expected_len = int(expected_len)
    vector_len = int(vector.shape[-1])

    if abs(vector_len - expected_len) > len_tolerance:
        # Ensure vector was close to expected length to begin with
        raise ValueError('Vector length: {} differs from expected length: {} '
                         'beyond tolerance of : {}'.format(vector_len,
                                                           expected_len,
                                                           len_tolerance))
    # Pick tensorflow or numpy.
    lib = tf if use_tf else np

    is_1d = (len(vector.shape) == 1)
    vector = vector[lib.newaxis, :] if is_1d else vector

    # Pad missing samples
    if vector_len < expected_len:
        n_padding = expected_len - vector_len
        vector = lib.pad(
            vector, ((0, 0), (0, n_padding)),
            mode='constant',
            constant_values=pad_value)
    # Trim samples
    elif vector_len > expected_len:
        vector = vector[..., :expected_len]

    # Remove temporary batch dimension.
    vector = vector[0] if is_1d else vector
    return vector


def hz_to_midi(frequencies: Number) -> Number:
    """TF-compatible hz_to_midi function."""
    frequencies = tf_float32(frequencies)
    notes = 12.0 * (logb(frequencies, 2.0) - logb(440.0, 2.0)) + 69.0
    # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
    notes = tf.where(tf.less_equal(frequencies, 0.0), 0.0, notes)
    return notes


"""Utilites for postprocessing datasets."""


def detect_notes(loudness_db,
                 f0_confidence,
                 note_threshold=1.0,
                 exponent=2.0,
                 smoothing=40,
                 f0_confidence_threshold=0.7,
                 min_db=-120.):
    """Detect note on-off using loudness and smoothed f0_confidence."""
    mean_db = np.mean(loudness_db)
    db = smooth(f0_confidence**exponent, smoothing) * (loudness_db - min_db)
    db_threshold = (mean_db - min_db) * f0_confidence_threshold**exponent
    note_on_ratio = db / db_threshold
    mask_on = note_on_ratio >= note_threshold
    return mask_on, note_on_ratio


def fit_quantile_transform(loudness_db, mask_on, inv_quantile=None):
    """Fits quantile normalization, given a note_on mask.

    Optionally, performs the inverse transformation given a pre-fitted transform.
    Args:
      loudness_db: Decibels, shape [batch, time]
      mask_on: A binary mask for when a note is present, shape [batch, time].
      inv_quantile: Optional pretrained QuantileTransformer to perform the inverse
        transformation.

    Returns:
      Trained quantile transform. Also returns the renormalized loudnesses if
        inv_quantile is provided.
    """
    quantile_transform = QuantileTransformer()
    loudness_flat = np.ravel(loudness_db[mask_on])[:, np.newaxis]
    loudness_flat_q = quantile_transform.fit_transform(loudness_flat)

    if inv_quantile is None:
        return quantile_transform
    else:
        loudness_flat_norm = inv_quantile.inverse_transform(loudness_flat_q)
        loudness_norm = np.ravel(np.copy(loudness_db))[:, np.newaxis]
        loudness_norm[mask_on] = loudness_flat_norm
        return quantile_transform, loudness_norm


class QuantileTransformer:
    """Transform features using quantiles information.

    Stripped down version of sklearn.preprocessing.QuantileTransformer.
    https://github.com/scikit-learn/scikit-learn/blob/
    863e58fcd5ce960b4af60362b44d4f33f08c0f97/sklearn/preprocessing/_data.py

    Putting directly in ddsp library to avoid dependency on sklearn that breaks
    when pickling and unpickling from different versions of sklearn.
    """

    def __init__(self,
                 n_quantiles=1000,
                 output_distribution='uniform',
                 subsample=int(1e5)):
        """Constructor.

        Args:
          n_quantiles: int, default=1000 or n_samples Number of quantiles to be
            computed. It corresponds to the number of landmarks used to discretize
            the cumulative distribution function. If n_quantiles is larger than the
            number of samples, n_quantiles is set to the number of samples as a
            larger number of quantiles does not give a better approximation of the
            cumulative distribution function estimator.
          output_distribution: {'uniform', 'normal'}, default='uniform' Marginal
            distribution for the transformed data. The choices are 'uniform'
            (default) or 'normal'.
          subsample: int, default=1e5 Maximum number of samples used to estimate
            the quantiles for computational efficiency. Note that the subsampling
            procedure may differ for value-identical sparse and dense matrices.
        """
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = np.random.mtrand._rand

    def _dense_fit(self, x, random_state):
        """Compute percentiles for dense matrices.

        Args:
          x: ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
          random_state: Numpy random number generator.
        """
        n_samples, _ = x.shape
        references = self.references_ * 100

        self.quantiles_ = []
        for col in x.T:
            if self.subsample < n_samples:
                subsample_idx = random_state.choice(
                    n_samples, size=self.subsample, replace=False)
                col = col.take(subsample_idx, mode='clip')
            self.quantiles_.append(np.nanpercentile(col, references))
        self.quantiles_ = np.transpose(self.quantiles_)
        # Due to floating-point precision error in `np.nanpercentile`,
        # make sure that quantiles are monotonically increasing.
        # Upstream issue in numpy:
        # https://github.com/numpy/numpy/issues/14685
        self.quantiles_ = np.maximum.accumulate(self.quantiles_)

    def fit(self, x):
        """Compute the quantiles used for transforming.

        Parameters
        ----------
        Args:
          x: {array-like, sparse matrix} of shape (n_samples, n_features)
            The data used to scale along the features axis. If a sparse
            matrix is provided, it will be converted into a sparse
            ``csc_matrix``. Additionally, the sparse matrix needs to be
            nonnegative if `ignore_implicit_zeros` is False.

        Returns:
          self: object
             Fitted transformer.
        """
        if self.n_quantiles <= 0:
            raise ValueError("Invalid value for 'n_quantiles': %d. "
                             'The number of quantiles must be at least one.' %
                             self.n_quantiles)
        n_samples = x.shape[0]
        self.n_quantiles_ = max(1, min(self.n_quantiles, n_samples))

        # Create the quantiles of reference
        self.references_ = np.linspace(0, 1, self.n_quantiles_, endpoint=True)
        self._dense_fit(x, self.random_state)
        return self

    def _transform_col(self, x_col, quantiles, inverse):
        """Private function to transform a single feature."""
        output_distribution = self.output_distribution
        bounds_threshold = 1e-7

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    x_col = stats.norm.cdf(x_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
            if output_distribution == 'normal':
                lower_bounds_idx = (x_col - bounds_threshold < lower_bound_x)
                upper_bounds_idx = (x_col + bounds_threshold > upper_bound_x)
            if output_distribution == 'uniform':
                lower_bounds_idx = (x_col == lower_bound_x)
                upper_bounds_idx = (x_col == upper_bound_x)

        isfinite_mask = ~np.isnan(x_col)
        x_col_finite = x_col[isfinite_mask]
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            x_col[isfinite_mask] = .5 * (
                np.interp(x_col_finite, quantiles, self.references_) -
                np.interp(-x_col_finite, -quantiles[::-1], -self.references_[::-1]))
        else:
            x_col[isfinite_mask] = np.interp(x_col_finite, self.references_,
                                             quantiles)

        x_col[upper_bounds_idx] = upper_bound_y
        x_col[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid='ignore'):  # hide NaN comparison warnings
                if output_distribution == 'normal':
                    x_col = stats.norm.ppf(x_col)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    clip_min = stats.norm.ppf(bounds_threshold - np.spacing(1))
                    clip_max = stats.norm.ppf(
                        1 - (bounds_threshold - np.spacing(1)))
                    x_col = np.clip(x_col, clip_min, clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let x_col unchanged

        return x_col

    def _transform(self, x, inverse=False):
        """Forward and inverse transform.

        Args:
          x : ndarray of shape (n_samples, n_features)
            The data used to scale along the features axis.
          inverse : bool, default=False
            If False, apply forward transform. If True, apply
            inverse transform.

        Returns:
          x : ndarray of shape (n_samples, n_features)
            Projected data
        """
        x = np.array(x)  # Explicit copy.
        for feature_idx in range(x.shape[1]):
            x[:, feature_idx] = self._transform_col(
                x[:, feature_idx], self.quantiles_[:, feature_idx], inverse)
        return x

    def transform(self, x):
        """Feature-wise transformation of the data."""
        return self._transform(x, inverse=False)

    def inverse_transform(self, x):
        """Back-projection to the original space."""
        return self._transform(x, inverse=True)

    def fit_transform(self, x):
        """Fit and transform."""
        return self.fit(x).transform(x)


def compute_dataset_statistics(data_provider,
                               batch_size=1,
                               power_frame_size=256):
    """Calculate dataset stats.

    Args:
      data_provider: A DataProvider from ddsp.training.data.
      batch_size: Iterate over dataset with this batch size.
      power_frame_size: Calculate power features on the fly with this frame size.

    Returns:
      Dictionary of dataset statistics. This is an overcomplete set of statistics,
      as there are now several different tone transfer implementations (js, colab,
      vst) that need different statistics for normalization.
    """
    print('Calculating dataset statistics for', data_provider)
    data_iter = iter(data_provider.get_batch(batch_size, repeats=1))

    # Unpack dataset.
    i = 0
    loudness = []
    power = []
    f0 = []
    f0_conf = []
    audio = []

    for batch in data_iter:
        loudness.append(batch['loudness_db'])
        power.append(
            spectral_ops.compute_power(batch['audio'], frame_size=power_frame_size))
        f0.append(batch['f0_hz'])
        f0_conf.append(batch['f0_confidence'])
        audio.append(batch['audio'])
        i += 1

    print(f'Computing statistics for {i * batch_size} examples.')

    loudness = np.vstack(loudness)
    power = np.vstack(power)
    f0 = np.vstack(f0)
    f0_conf = np.vstack(f0_conf)
    audio = np.vstack(audio)

    # Fit the transform.
    trim_end = 20
    f0_trimmed = f0[:, :-trim_end]
    pitch_trimmed = hz_to_midi(f0_trimmed)
    power_trimmed = power[:, :-trim_end]
    loudness_trimmed = loudness[:, :-trim_end]
    f0_conf_trimmed = f0_conf[:, :-trim_end]

    # Detect notes.
    mask_on, _ = detect_notes(loudness_trimmed, f0_conf_trimmed)
    quantile_transform = fit_quantile_transform(loudness_trimmed, mask_on)

    # Pitch statistics.
    def get_stats(x, prefix='x', note_mask=None):
        if note_mask is None:
            mean_max = np.mean(np.max(x, axis=-1))
            mean_min = np.mean(np.min(x, axis=-1))
        else:
            max_list = []
            for x_i, m in zip(x, note_mask):
                if np.sum(m) > 0:
                    max_list.append(np.max(x_i[m]))
            mean_max = np.mean(max_list)

            min_list = []
            for x_i, m in zip(x, note_mask):
                if np.sum(m) > 0:
                    min_list.append(np.min(x_i[m]))
            mean_min = np.mean(min_list)

            x = x[note_mask]

        return {
            f'mean_{prefix}': np.mean(x),
            f'max_{prefix}': np.max(x),
            f'min_{prefix}': np.min(x),
            f'mean_max_{prefix}': mean_max,
            f'mean_min_{prefix}': mean_min,
            f'std_{prefix}': np.std(x)
        }

    ds_stats = {}
    ds_stats.update(get_stats(pitch_trimmed, 'pitch'))
    ds_stats.update(get_stats(power_trimmed, 'power'))
    ds_stats.update(get_stats(loudness_trimmed, 'loudness'))
    ds_stats.update(get_stats(pitch_trimmed, 'pitch_note', mask_on))
    ds_stats.update(get_stats(power_trimmed, 'power_note', mask_on))
    ds_stats.update(get_stats(loudness_trimmed, 'loudness_note', mask_on))

    ds_stats['quantile_transform'] = quantile_transform
    return ds_stats

# ------------------------------------------------------------------------------
# Loudness Normalization
# ------------------------------------------------------------------------------


def smooth(x, filter_size=3):
    """Smooth 1-d signal with a box filter."""
    x = tf.convert_to_tensor(x, tf.float32)
    is_2d = len(x.shape) == 2
    x = x[:, :, tf.newaxis] if is_2d else x[tf.newaxis, :, tf.newaxis]
    w = tf.ones([filter_size])[:, tf.newaxis, tf.newaxis] / float(filter_size)
    y = tf.nn.conv1d(x, w, stride=1, padding='SAME')
    y = y[:, :, 0] if is_2d else y[0, :, 0]
    return y.numpy()

# /////////////////////////////////////////////////////////////////////


fs = RATE
x, fs = librosa.load('piano_sev.mp3', sr=fs, mono=True)
os.system('afplay piano_sev.mp3')

print('\nExtracting audio features...')

# Setup the session.
reset_crepe()
print('Successfully reset crepe')
np_config.enable_numpy_behavior()

# Compute features.
start_time = time.time()
print('Successfully set start time')
audio_features = compute_audio_features(x)
print('Successfully computed audio features')

audio_features['loudness_db'] = audio_features['loudness_db'].astype(
    np.float32)
audio_features_mod = None
print('\n\nAudio features took %.1f seconds\n\n' % (time.time() - start_time))

# /////////////////////////////////////////////////////////////////////

# Load a model

model = 'Tenor_Saxophone'
MODEL = model

model_dir = 'pretrained'
gin_file = os.path.join(model_dir, 'operative_config-0.gin')

# Load the dataset statistics.
DATASET_STATS = None
dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
print(f'Loading dataset statistics from {dataset_stats_file}')
try:
    if tf.io.gfile.exists(dataset_stats_file):
        with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
            DATASET_STATS = pickle.load(f)
except Exception as err:
    print('Loading dataset statistics from pickle failed: {}.'.format(err))

# Parse gin config,
with gin.unlock_config():
    gin.parse_config_file(gin_file, skip_unknown=True)

# Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
ckpt_name = ckpt_files[0].split('.')[0]
ckpt = os.path.join(model_dir, ckpt_name)

# Ensure dimensions and sampling rates are equal
time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
n_samples_train = gin.query_parameter('Harmonic.n_samples')
hop_size = int(n_samples_train / time_steps_train)

time_steps = int(x.shape[0] / hop_size)
n_samples = time_steps * hop_size

print("===Trained model===")
print("Time Steps", time_steps_train)
print("Samples", n_samples_train)
print("Hop Size", hop_size)
print("\n===Resynthesis===")
print("Time Steps", time_steps)
print("Samples", n_samples)
print('')

gin_params = [
    'Harmonic.n_samples = {}'.format(n_samples),
    'FilteredNoise.n_samples = {}'.format(n_samples),
    'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
    # Avoids cumsum accumulation errors.
    'oscillator_bank.use_angular_cumsum = True',
]

with gin.unlock_config():
    gin.parse_config(gin_params)

# Trim all input vectors to correct lengths
for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
    audio_features[key] = audio_features[key][:time_steps]
audio_features['audio'] = audio_features['audio'][:n_samples]

# Set up the model just to predict audio given new conditioning
model = ddsp.training.models.Autoencoder()
model.restore(ckpt)

# Build model by running a batch through it.
start_time = time.time()
_ = model(audio_features, training=False)
print('Restoring model took %.1f seconds' % (time.time() - start_time))

# /////////////////////////////////////////////////////////////////////

# Modify conditioning

# Note Detection
# leave this at 1.0 for most cases
threshold = 1  # min: 0.0, max:2.0, step:0.01

# Automatic
ADJUST = True

# Quiet parts without notes detected (dB)
quiet = 20  # @param {type:"slider", min: 0, max:60, step:1}

# Force pitch to nearest note (amount)
autotune = 0  # @param {type:"slider", min: 0.0, max:1.0, step:0.1}

# Manual

# Shift the pitch (octaves)
pitch_shift = 0  # @param {type:"slider", min:-2, max:2, step:1}

# Adjust the overall loudness (dB)
loudness_shift = 0  # @param {type:"slider", min:-20, max:20, step:1}

audio_features_mod = {k: np.copy(v) for k, v in audio_features.items()}

# Helper functions.


def shift_ld(audio_features, ld_shift=0.0):
    """Shift loudness by a number of ocatves."""
    audio_features['loudness_db'] += ld_shift
    return audio_features


def shift_f0(audio_features, pitch_shift=0.0):
    """Shift f0 by a number of ocatves."""
    audio_features['f0_hz'] *= 2.0 ** (pitch_shift)
    audio_features['f0_hz'] = np.clip(audio_features['f0_hz'],
                                      0.0,
                                      librosa.midi_to_hz(110.0))
    return audio_features


mask_on = None

# /////////////////////////////////////////////////////////////////////

if ADJUST and DATASET_STATS is not None:
    # Detect sections that are "on".
    mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                          audio_features['f0_confidence'],
                                          threshold)

    if np.any(mask_on):
        # Shift the pitch register.
        target_mean_pitch = DATASET_STATS['mean_pitch']
        pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
        mean_pitch = np.mean(pitch[mask_on])
        p_diff = target_mean_pitch - mean_pitch
        p_diff_octave = p_diff / 12.0
        round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
        p_diff_octave = round_fn(p_diff_octave)
        audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)

        # Quantile shift the note_on parts.
        _, loudness_norm = fit_quantile_transform(
            audio_features['loudness_db'],
            mask_on,
            inv_quantile=DATASET_STATS['quantile_transform'])

        # Turn down the note_off parts.
        mask_off = np.logical_not(mask_on)
        loudness_norm[mask_off] -= quiet * \
            (1.0 - note_on_value[mask_off][:, np.newaxis])
        loudness_norm = np.reshape(
            loudness_norm, audio_features['loudness_db'].shape)

        audio_features_mod['loudness_db'] = loudness_norm

        # Auto-tune.
        if autotune:
            f0_midi = np.array(ddsp.core.hz_to_midi(
                audio_features_mod['f0_hz']))
            tuning_factor = get_tuning_factor(
                f0_midi, audio_features_mod['f0_confidence'], mask_on)
            f0_midi_at = auto_tune(
                f0_midi, tuning_factor, mask_on, amount=autotune)
            audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)

        else:
            print('\nSkipping auto-adjust (no notes detected or ADJUST box empty).')

else:
    print('\nSkipping auto-adujst (box not checked or no dataset statistics found).')

# Manual Shifts.
audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
audio_features_mod = shift_f0(audio_features_mod, pitch_shift)

# /////////////////////////////////////////////////////////////////////

# Resynthesize Audio

af = audio_features if audio_features_mod is None else audio_features_mod

# Run a batch of predictions.
start_time = time.time()
outputs = model(af, training=False)
audio_gen = model.get_audio_from_outputs(outputs)
print('Prediction took %.1f seconds' % (time.time() - start_time))

print('\nResynthesis\n\n\n')
sf.write('sax_sev.wav', audio_gen.T, fs)
os.system('afplay sax_sev.wav')
os.remove('sax_sev.wav')
quit()
