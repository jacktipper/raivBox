import collections
import copy
# from typing import Any, Dict, Optional, Sequence, Text, TypeVar

import gin
import numpy as np
from scipy import fftpack
import tensorflow.compat.v2 as tf



def tf_float32(x):
  """Ensure array/tensor is a float32 tf.Tensor."""
  if isinstance(x, tf.Tensor):
    return tf.cast(x, dtype=tf.float32)  # This is a no-op if x is float32.
  else:
    return tf.convert_to_tensor(x, tf.float32)


def safe_log(x, eps=1e-5):
  """Avoid taking the log of a non-positive number."""
  safe_x = tf.where(x <= eps, eps, x)
  return tf.math.log(safe_x)



