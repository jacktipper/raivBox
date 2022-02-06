# Hyperparameters

RATE = 16000  # Hz
INPUT_PATH = 'audio/input.wav'
OUTPUT_PATH = 'audio/output.wav'

VOICING_THRESHOLD = -125  # dB
PITCH_SHIFT = -1  # octaves up or down
LOUDNESS = 5  # loudness shift (normalized anyway)


# Libraries
import os
os.system('./init-py.sh')

import warnings
warnings.filterwarnings("ignore")

import time
import Jetson.GPIO as GPIO
from multiprocessing import Process

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
led0pin = 20
GPIO.setup(led0pin, GPIO.OUT, initial=GPIO.LOW)

def blinker():
    try:
        while True:
            GPIO.output(led0pin, GPIO.HIGH)
            time.sleep(0.08)
            GPIO.output(led0pin, GPIO.LOW)
            time.sleep(0.12)
    finally:
        GPIO.output(led0pin, GPIO.LOW)
    return

blink = Process(target=blinker)
blink.start()

import gin
import librosa
import pickle
import numpy as np
import soundfile as sf
import tensorflow.compat.v2 as tf
from ddsp.core import make_iterable, hz_to_midi, copy_if_tf_function
from ddsp.training.models import Autoencoder
from tensorflow.python.ops.numpy_ops import np_config
from datetime import datetime


# Disable GPU (CUDA and TensorFlow need to be patched)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# DDSP Helper Functions (extracted from official modules)

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
                           frame_rate=250):
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
USE_TF = False  # tensorflow or numpy?

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


def power_to_db(power, ref_db=0.0, range_db=LD_RANGE, use_tf=USE_TF):
  """Converts power from linear scale to decibels."""
  # Choose library.
  maximum = tf.maximum if use_tf else np.maximum
  log_base10 = log10 if use_tf else np.log10

  # Convert to decibels.
  pmin = 10**-(range_db / 10.0)
  power = maximum(pmin, power)
  db = 10.0 * log_base10(power)

  # Set dynamic range.
  db -= ref_db
  db = maximum(db, -range_db)
  return db


def compute_loudness(audio, # this function has been updated :)
                     sample_rate=RATE,
                     frame_rate=250,
                     n_fft=2048,
                     range_db=LD_RANGE,
                     ref_db=0.0,
                     use_tf=USE_TF,
                     pad_end=True):
    """Perceptual loudness (weighted power) in dB.
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
        (A_weighting + 10 * log10(abs(stft(audio))**2.0). The old (<v2.0.0)
        default value corresponded to white noise with amplitude=1.0 and
        n_fft=2048. With v2.0.0 it was set to 0.0 to be more consistent with power
        calculations that have a natural scale for 0 dB being amplitude=1.0.
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
    reduce_mean = tf.reduce_mean if use_tf else np.mean
    stft_fn = stft if use_tf else stft_np

    # Make inputs tensors for tensorflow.
    audio = tf_float32(audio) if use_tf else audio

    # Temporarily a batch dimension for single examples.
    is_1d = (len(audio.shape) == 1)
    audio = audio[lib.newaxis, :] if is_1d else audio

    # Take STFT.
    hop_size = sample_rate // frame_rate
    overlap = 1 - hop_size / n_fft
    s = stft_fn(audio, frame_size=n_fft, overlap=overlap, pad_end=pad_end)

    # Compute power.
    amplitude = lib.abs(s)
    power = amplitude**2

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]

    # Perform weighting in linear scale, a_weighting given in decibels.
    weighting = 10**(a_weighting/10)
    power = power * weighting

    # Average over frequencies (weighted power per a bin).
    avg_power = reduce_mean(power, axis=-1)
    loudness = power_to_db(avg_power,
                           ref_db=ref_db,
                           range_db=range_db,
                           use_tf=use_tf)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    # Compute expected length of loudness vector.
    expected_secs = audio.shape[-1] / float(sample_rate)
    expected_len = int(expected_secs * frame_rate)

    # Pad with `-range_db` noise floor or trim vector.
    loudness = pad_or_trim_to_expected_length(
        loudness, expected_len, -range_db, use_tf=use_tf)

    return loudness


def compute_f0(audio, sample_rate=RATE, frame_rate=250, viterbi=True):
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
    c_step_size = 1000 / frame_rate  # milliseconds
    frame_length = int(c_step_size*(sample_rate/frame_rate))
    expected_len = int(n_secs * frame_rate)
    audio = np.asarray(audio)

    f0_hz = librosa.yin(audio,
                        fmin=60,
                        fmax=2100,
                        sr=sample_rate,
                        frame_length=frame_length,
                        center=False)
    f0_confidence = np.ones(len(f0_hz))

    # Postprocessing on f0_hz
    f0_hz = pad_or_trim_to_expected_length(
        f0_hz, expected_len, 0)  # pad with 0
    f0_hz = f0_hz.astype(np.float32)

    # Postprocessing on f0_confidence
    f0_confidence = pad_or_trim_to_expected_length(
        f0_confidence, expected_len, 1)

    return f0_hz, f0_confidence


def pad_or_trim_to_expected_length(vector,
                                   expected_len,
                                   pad_value=0,
                                   len_tolerance=20,
                                   use_tf=USE_TF):
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


# Helper functions
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


def smooth(x, filter_size=3):
    """Smooth 1-d signal with a box filter."""
    x = tf.convert_to_tensor(x, tf.float32)
    is_2d = len(x.shape) == 2
    x = x[:, :, tf.newaxis] if is_2d else x[tf.newaxis, :, tf.newaxis]
    w = tf.ones([filter_size])[:, tf.newaxis, tf.newaxis] / float(filter_size)
    y = tf.nn.conv1d(x, w, stride=1, padding='SAME')
    y = y[:, :, 0] if is_2d else y[0, :, 0]
    return y.numpy()


blink.terminate()
blink = None
GPIO.output(led0pin, GPIO.HIGH)
ready = 'flags/read.y'

try:
    while True:
        if os.path.exists(ready):

            blink = Process(target=blinker)
            blink.start()
            
            # Audio Feature Extraction
            fs = RATE
            x, fs = librosa.load(INPUT_PATH, sr=fs, mono=True)


            # Compute features.
            audio_features = compute_audio_features(x)

            np_config.enable_numpy_behavior()

            audio_features['loudness_db'] = audio_features['loudness_db'].astype(
                np.float32)
            audio_features_mod = None


            # Fixing f0_confidence
            audio_features['f0_confidence'] = np.ones(len(audio_features['f0_confidence']))

            voicing_threshold = VOICING_THRESHOLD  # dB

            for i in range(len(audio_features['f0_confidence'])):
                if audio_features['loudness_db'][i] < voicing_threshold:
                    audio_features['f0_confidence'][i] = 0.

            audio_features['f0_confidence'] = audio_features['f0_confidence'].astype(
                np.float32)


            # Load a model
            MODEL = open('models/model.txt').read()[0:-1]
            print('Model Loaded:', MODEL)
            model_dir = 'models/{}'.format(MODEL)
            gin_file = os.path.join(model_dir, 'operative_config-0.gin')

            # Load the dataset statistics.
            DATASET_STATS = None
            dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
            try:
                if tf.io.gfile.exists(dataset_stats_file):
                    with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                        DATASET_STATS = pickle.load(f)
            except Exception as err:
                print('Loading dataset statistics from pickle failed: {}.'.format(err))

            # Parse gin config,
            with gin.unlock_config():
                gin.parse_config_file(gin_file, skip_unknown=True)

            # Ensure dimensions and sampling rates are equal
            time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
            n_samples_train = gin.query_parameter('Harmonic.n_samples')
            hop_size = int(n_samples_train / time_steps_train)

            time_steps = int(x.shape[0] / hop_size)
            n_samples = time_steps * hop_size

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


            # Modify conditioning

            # Note Detection (leave this at 1.0 for most cases)
            threshold = 1  # min: 0.0, max:2.0, step:0.01

            # Trim
            TRIM = -15

            # Automatic
            ADJUST = True

            # Quiet parts without notes detected (dB)
            quiet = 0  # @param {type:"slider", min: 0, max:60, step:1}

            # Shift the pitch (octaves)
            pitch_shift = PITCH_SHIFT  # @param {type:"slider", min:-2, max:2, step:1}

            # @markdown Adjust the overall loudness (dB)
            loudness_shift = LOUDNESS  # @param {type:"slider", min:-20, max:20, step:1}

            audio_features_mod = {k: np.copy(v) for k, v in audio_features.items()}

            # Adjustments
            mask_on = None

            if ADJUST and DATASET_STATS is not None:
                # Detect sections that are "on".
                mask_on, note_on_value = detect_notes(audio_features['loudness_db'],
                                                    audio_features['f0_confidence'],
                                                    threshold)

                if np.any(mask_on):
                    # Shift the pitch register.
                    target_mean_pitch = DATASET_STATS['mean_pitch']
                    pitch = hz_to_midi(audio_features['f0_hz'])
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

            else:
                print('\nSkipping auto-adujst (no dataset statistics found).')

            # Manual Shifts.
            audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
            audio_features_mod = shift_f0(audio_features_mod, pitch_shift)
            af = audio_features if audio_features_mod is None else audio_features_mod


            # Load the model
            model = Autoencoder()
            model.load_weights('models/{}/model_data/'.format(MODEL))


            # Run inference
            outputs = model(af, training=False)

            audio_gen = model.get_audio_from_outputs(outputs)
            audio_gen = audio_gen.numpy()[0]
            audio_gen = audio_gen/np.max(audio_gen)

            dest = str(datetime.now())[0:19]
            if os.path.exists(OUTPUT_PATH):
                os.rename(OUTPUT_PATH, str('audio/archive/out_' + dest + '.wav'))

            sf.write(OUTPUT_PATH, audio_gen, fs)
            print('\n    Output Successfully Synthesized\n')

            blink.terminate()
            blink = None
            GPIO.output(led0pin, GPIO.HIGH)

            os.rename(INPUT_PATH, str('audio/archive/in_' + dest + '.wav'))
            os.remove(ready)
        
        # set the status check interval for the while loop (responsivity)
        time.sleep(0.02)

finally:
    if os.path.exists(ready):
        os.remove(ready)
    dest = str(datetime.now())[0:19]
    if os.path.exists(INPUT_PATH): 
        os.rename(INPUT_PATH, str('audio/archive/f_in_' + dest + '.wav'))
    if os.path.exists(OUTPUT_PATH):
        os.rename(OUTPUT_PATH, str('audio/archive/f_out_' + dest + '.wav'))
    GPIO.output(led0pin, GPIO.LOW)