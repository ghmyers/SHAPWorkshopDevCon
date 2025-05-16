import numpy as np
import random
import pandas as pd
def train_test_split_evenSites(df, split_pct, seed):
    # Set random seed
    np.random.seed(seed)
    random.seed(seed)

    # Get unique site identifiers
    sites = df['STAID'].unique()

    # Store splits here
    train_splits = []
    test_splits  = []

    for site in sites:
        temp_df = df[df['STAID'] == site]
        split_ind = int(np.floor((1-split_pct)*int(len(temp_df))))
        start_ind = np.random.randint(0, len(temp_df) - split_ind)
        end_ind = start_ind + split_ind
        test_df = temp_df.iloc[start_ind:end_ind, :]
        train_df = pd.concat([temp_df.iloc[:start_ind, :], temp_df.iloc[end_ind:, :]])
        train_splits.append(train_df)
        test_splits.append(test_df)

    # Zip lists together, shuffle them, then unzip them
    zipped_list = list(zip(train_splits, test_splits))
    random.shuffle(zipped_list)
    train_splits, test_splits = zip(*zipped_list)

    Train = pd.concat(train_splits)
    Test = pd.concat(test_splits)

    train_sta = Train['STAID'].copy()
    val_sta   = Test['STAID'].copy()
    Train.drop("STAID", axis=1, inplace=True)
    Test.drop('STAID', axis=1, inplace=True)

    X_train = Train.drop('Q', axis=1)
    y_train = Train['Q']
    X_test = Test.drop('Q', axis=1)
    y_test = Test['Q']

    return X_train, y_train, X_test, y_test, train_sta, val_sta

import tensorflow.keras as keras
import tensorflow as tf


def make_dataset(X, y, staid, window, batch, buffer=180, shuffle=True):

    site_ds = []
    for site in np.unique(staid):
        mask   = staid == site
        x_site = X[mask].to_numpy(np.float32)
        y_site = y[mask].to_numpy(np.float32)

        # Build a dataset of sliding windows
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                x_site, y_site,
                sequence_length=window,
                sampling_rate=1,
                batch_size=batch,
                shuffle=False)
        site_ds.append(ds)

    # Concatenate sliding windows 1 x 1
    full_ds = site_ds[0]
    for ds in site_ds[1:]:
        full_ds = full_ds.concatenate(ds)

    # Store full dataset in RAM
    full_ds = full_ds.cache()

    # Shuffle dataset
    if shuffle:
        full_ds = full_ds.shuffle(buffer, seed=42, reshuffle_each_iteration=True)

    return full_ds.prefetch(tf.data.AUTOTUNE)

from tensorflow.keras import layers, metrics

def build_lstm(window_length: int,
               n_features:   int,
               hidden_units: int   = 32,
               dropout:      float = 0.15,
               lr:           float = 1e-3):
    """Return a compiled 1-layer LSTM regression model."""

    inputs  = keras.Input((window_length, n_features))
    x       = layers.LSTM(hidden_units)(inputs)
    x       = layers.Dropout(dropout)(x)
    x       = layers.Dense(hidden_units, activation="relu")(x)
    outputs = layers.Dense(1)(x)                    # **linear head**

    model = keras.Model(inputs, outputs, name="lstm_regressor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1),
        loss="mse",
        metrics=[keras.metrics.RootMeanSquaredError(name="RMSE"),
                 keras.metrics.MeanAbsoluteError(name="MAE")]
    )
    model.summary()
    return model

def reshape_data(X, y, window_length):
    # Convert X and y to numpy arrays
    X_array = X.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)
    
    n_samples = len(X)
    n_sequences = n_samples - window_length
    
    # Lists to hold the valid sequences
    valid_sequences = []
    valid_targets = []
    target_indices = []
    
    indices = np.arange(n_sequences)
    
    for i in indices:
        # # Get start and end dates of sequence
        # sequence_start = X.index[i]
        # sequence_end = X.index[i + window_length]
        
        X_seq = X_array[i:i + window_length]
        y_seq = y_array[i + window_length]
        
        valid_sequences.append(X_seq)
        valid_targets.append(y_seq)
        
        # Store the corresponding indices for the sequence and the target
        target_indices.append(X.index[i + window_length])
    

    # Convert lists to numpy arrays
    X = np.array(valid_sequences)
    y = y[target_indices]
    
    return X, y

import numpy as np

import numpy as np
from numpy.lib import stride_tricks

def build_lstm_windows(X, y, window, stride: int = 1, drop_remainder: bool = True):
    """
    Turn a time-ordered feature matrix X and target vector y into
    (n_windows, window, n_features) + (n_windows,) arrays for an LSTM.

    Compatible with older NumPy versions that lack sliding_window_view.
    """
    # ---------- basic checks ----------
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if window < 1 or stride < 1:
        raise ValueError("`window` and `stride` must be positive integers")

    n_rows, n_feat = X.shape
    n_win = (n_rows - window) // stride + 1
    if n_win <= 0:
        raise ValueError("`window` is longer than the series")

    # ---------- try fast helper (NumPy ≥ 1.20) ----------
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        X_windows = sliding_window_view(X, (window, n_feat))[::stride, 0]
        y_windows = sliding_window_view(y, window)[::stride, -1]

    # ---------- fallback for older NumPy ----------
    except ImportError:
        # Build strided view manually
        win_stride = X.strides[0]
        new_shape  = (n_win, window, n_feat)
        new_strides = (stride * win_stride, win_stride, X.strides[1])
        X_view = stride_tricks.as_strided(X, shape=new_shape, strides=new_strides)

        y_shape   = (n_win, window)
        y_strides = (stride * y.strides[0], y.strides[0])
        y_view    = stride_tricks.as_strided(y, shape=y_shape, strides=y_strides)

        # Copy to make them safe if the originals are mutated later
        X_windows = X_view.copy()
        y_windows = y_view[:, -1].copy()

    # -------- drop remainder handling (optional) --------
    if not drop_remainder and (n_rows - window) % stride:
        # one extra, shorter window at the tail
        start = n_rows - window + 1
        extra_X = X[start:]
        extra_y = y[start + window - 1]
        X_windows = np.concatenate([X_windows, extra_X[None, ...]], axis=0)
        y_windows = np.concatenate([y_windows, np.asarray([extra_y], dtype=np.float32)], axis=0)

    return X_windows, y_windows
