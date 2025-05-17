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


import numpy as np, h5py, tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.keras import layers
# from src.model_building import build_lstm
import shap

def get_lstm_shap_vals(weights_path: str, hidden_units: int, window_length:int, n_cols:int, n_batches:int, X_rs, feat_scaler, 
                       y_scaler, shap_outpath: str, basevals_outpath: str, shap_data_outpath: str):
    """Build LSTM model, loads in pretrained weights, calculates SHAP values for model, and saves SHAP values

    Args:
        weights_path (str): path to pretrained weights
        hidden_units (int): number of hidden units in LSTM
        n_batches (int): number of batches to use for shap value calculation
        window_length (int): window length for LSTM
        n_cols (int): number of features
        X_rs (numpy array): feature matrix
        feat_scaler (sklearn scaler): feature scaler
        y_scaler (sklearn scaler): target scaler
        shap_outpath (str): path to save SHAP values
        basevals_outpath (str): path to save base values from SHAP
        shap_data_outpath (str): path to save SHAP data
    """
    tf.compat.v1.disable_eager_execution()
    
    with h5py.File(weights_path, "r") as f, tf.compat.v1.Session() as sess:
        K.set_session(sess)

        model = build_lstm(window_length, n_cols, hidden_units=hidden_units)

        # locate layers by type 
        lstm_layer   = next(l for l in model.layers if isinstance(l, layers.LSTM))
        dense_hid    = next(l for l in model.layers
                            if isinstance(l, layers.Dense) and l.units == hidden_units)
        dense_out    = next(l for l in model.layers
                            if isinstance(l, layers.Dense) and l.units == 1)

        mw = f["model_weights"]

        # ----- LSTM weights (kernel, recurrent_kernel, bias) -----
        g_lstm = mw["lstm"]["lstm"]["lstm_cell"]
        lstm_w = [np.asarray(g_lstm[n][()], dtype=np.float32)
                for n in ("kernel", "recurrent_kernel", "bias")]
        lstm_layer.set_weights(lstm_w)

        # ----- Hidden dense weights (kernel, bias) -----
        g_dense = mw["dense"]["dense"]
        dense_w = [np.asarray(g_dense[n][()], dtype=np.float32)
                for n in ("kernel", "bias")]
        dense_hid.set_weights(dense_w)

        # ----- Output dense weights (kernel, bias) -----
        g_out = mw["dense_1"]["dense_1"]
        out_w = [np.asarray(g_out[n][()], dtype=np.float32)
                for n in ("kernel", "bias")]
        dense_out.set_weights(out_w)

        # Get 1000 random batches
        rng   = np.random.default_rng(42)          
        idx   = rng.choice(len(X_rs), 1000, replace=False)
        X_shap_lstm = X_rs[idx]
        explainer = shap.DeepExplainer(model, X_shap_lstm, session=sess)
        shap_vals = explainer.shap_values(X_shap_lstm)
        
        shap_explainer = shap.Explanation(values=shap_vals, base_values=explainer.expected_value, data=X_shap_lstm)

        # Flatten shap values along the second dimension to turn (n_batches, window_length, n_features) array into (n_batches * window_length, n_features) array
        shap_values_2d = shap_explainer[0].values.reshape(-1, shap_explainer[0].values.shape[-1]) 
        shap_base_values_2d = shap_explainer
        data_2d = shap_explainer.data.reshape(-1, shap_explainer.data.shape[-1])
        
        # Get scaler values
        y_scaler_mean = y_scaler.mean_[0]
        y_scaler_scale = y_scaler.scale_[0]
        
        # Multiply by the scaler values
        shap_values_2d *= y_scaler_scale
        shap_explainer.base_values *= y_scaler_scale

        # Add the mean to the base values (model mean) to rescale
        shap_explainer.base_values += y_scaler_mean
        base_values_2d = shap_explainer.base_values

        # Rescale the input data
        data_2d = feat_scaler.inverse_transform(data_2d)
        
        # Save transformed data
        np.savez(shap_data_outpath, data_2d=data_2d)
        np.savez(basevals_outpath, base_values_2d=base_values_2d)
        np.savez(shap_outpath, shap_values_2d=shap_values_2d)
                
        return shap_values_2d, base_values_2d, data_2d
    
def load_npz_array(path, key=None):
    """Return the first array in a .npz, or `key` if given."""
    obj = np.load(path, allow_pickle=True)   # returns ndarray or NpzFile
    if isinstance(obj, np.ndarray):          # .npy disguised as .npz ?
        return obj
    if key is None:                          # grab the only/first member
        key = obj.files[0]                   # e.g. 'arr_0'
    return obj[key]