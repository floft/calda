#!/usr/bin/env python3
"""
Synthetic datasets

Based on:
https://github.com/floft/codats/blob/v1/datasets/generate_trivial_datasets.py

Also see:
https://github.com/floft/codats/blob/v1/datasets/datasets.py

Usage:
    python -m datasets.synthetic_datasets
    ls datasets/synthetic/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linear(m, b, length=100, minvalue=0, maxvalue=2):
    x = np.arange(minvalue, maxvalue, (maxvalue-minvalue)/length).reshape(-1, 1).astype(np.float32)
    y = m*x + b
    return x, y


def display_xy(x, y, show=True):
    plt.figure()
    for i in range(y.shape[1]):
        plt.plot(x, y[:, i])

    if show:
        plt.show()


def to_pandas(y, labels):
    """
    Note: y is y-axis but actually the data, i.e. "x" in (x,y) ML terminology
    """
    df = pd.DataFrame(y.T)
    df.insert(0, 'class', pd.Series(np.squeeze(labels).astype(np.int32)+1, index=df.index))
    return df


def generate_positive_slope_data(n, display=False, add_noise=False,
        bmin=0.0, bmax=5.0, mmin=-1.0, mmax=1.0):
    """ Positive or negative slope lines """
    m = np.random.uniform(mmin, mmax, (1, n))
    b = np.random.uniform(bmin, bmax, (1, n))
    x, y = linear(m, b)
    labels = m > 0

    if add_noise:
        noise = np.random.normal(0.0, 0.25, (y.shape[0], n))
        y += noise

    if display:
        display_xy(x, y)

    return y, labels


def sine(m=1.0, b=0.0, f=None, amps=None, freq_noise=1.0, phase_shift=5.0,
        length=100, mint=0, maxt=2):
    """
    Generate a single or multiple sine waves (multiple if f is list)

    m - scale horizontally
    b - offset vertically
    f - either one frequency, a list of frequencies for each example (see below)
    amps - if specified, the amplitude of each specified frequency
    freq_noise - if not None, frequencies noisy about what is given in f
    phase_shift - how much to randomly offset in time (per frequency, so
        even with freq_noise=None, the two samples of f=[[1,2], [1,2]] will
        look different)
    length - number of samples to generate between mint and maxt
    mint - starting time
    maxt - stopping time

    100-length samples, one of 1 Hz and one of 2 Hz:
        sine(f=[[1], [2]], maxt=1, length=100, freq_noise=None, phase_shift=None)

    One 100-length sample with 1 and 2 Hz frequency components:
        sine(f=[[1,2]], maxt=1, length=100, freq_noise=None, phase_shift=None)

    Same frequency but different phase:
        sine(f=[[1]]*5, maxt=1, length=100, freq_noise=None, phase_shift=1.0)
    """
    multi_freq = isinstance(f, list) or isinstance(f, np.ndarray)

    # Set frequency if desired
    if f is None:
        s = np.array(1.0, dtype=np.float32)
        amps = np.array(1.0, dtype=np.float32)
    else:
        if amps is not None:
            amps = np.array(amps, dtype=np.float32)
        else:
            amps = np.array([1.0]*len(f), dtype=np.float32)

        f = np.array(f, dtype=np.float32)

        if freq_noise is not None:
            f += np.random.normal(0.0, freq_noise, f.shape)

        s = 2.0*np.pi*f

    x_orig = np.arange(mint, maxt, (maxt-mint)/length).reshape(-1, 1)
    x = x_orig

    if multi_freq:
        x_tile = np.tile(x, f.shape[0])  # e.g. (100,1) to (100,3) if 100 time steps, 3 frequencies
        x_newdim = np.expand_dims(x_tile, 2)  # now (100,3,1) so broadcasting works below
        x = x_newdim

    if phase_shift is None:
        phase_shift = 0
    else:
        if f is None:
            if isinstance(m, np.ndarray):
                shape = m.shape
            elif isinstance(b, np.ndarray):
                shape = b.shape
            else:
                raise NotImplementedError("When using phase shift one of m, b, "
                    "or f must be np.ndarray")
        else:
            # shape = (num_freq, num_examples)
            shape = f.shape

        phase_shift = np.random.normal(0.0, phase_shift, shape)

    y = m*amps*np.sin(s*(x+phase_shift))

    # Sum the extra dimension for multiple frequencies, e.g. from (100,3,2)
    # back to (100,3) if 2 frequencies each
    #
    # Note: make sure we don't add b in y above before summing, since otherwise
    # it'll be shifted much more than b
    if multi_freq:
        y = np.sum(y, axis=2) + b
    else:
        y += b

    return x_orig, y


def generate_positive_sine_data(n, display=False, add_noise=False,
        bmin=0.0, bmax=5.0, mmin=-1.0, mmax=1.0, f=None):
    """ Sine wave multiplied by positive or negative number and offset some """
    m = np.random.uniform(mmin, mmax, (1, n))
    b = np.random.uniform(bmin, bmax, (1, n))
    x, y = sine(m=m, b=b, f=f, freq_noise=None, phase_shift=None)
    labels = m > 0

    if add_noise:
        noise = np.random.normal(0.0, 0.1, (y.shape[0], n))
        y += noise

    if display:
        display_xy(x, y)

    return to_pandas(y, labels)


def generate_freq(n, display=False, amp_noise=False, freq_noise=False,
        fmin=1.0, fmax=2.0):
    """ Sine wave multiplied by positive or negative number and offset some
    Warning: probably recall Nyquist when setting fmax
    """
    freq = np.random.uniform(fmin, fmax, (n, 1))

    if freq_noise:
        freq += np.random.normal(0.0, 1.0, (n, 1))  # on order of freq diffs

    x, y = sine(f=freq, maxt=2, length=2*50)
    labels = freq > (fmax-fmin)/2

    if amp_noise:
        y += np.random.normal(0.0, 0.1, (y.shape[0], n))

    if display:
        display_xy(x, y)

    return y, labels


def generate_multi_freq(n, pos_f_all, neg_f_all,
        pos_amp_all=None, neg_amp_all=None,
        display=False,
        amp_noise=0.1, freq_noise=1.0, phase_shift=5.0,
        sample_freq=50, duration=2, b=0.0):
    """
    Generate data with different sets of frequencies for +/- classes

    Optionally specify different amplitudes for each of the frequencies. If not,
    then they all have the same amplitude (possibly with amplitude noise if
    amp_noise is not None).

    Note: {pos,neg}_{f,amp} is a 2D list to split frequencies/amplitudes across
    multiple channels. If you only want one channel, then do
        generate_multi_freq(x, [run_f], [walk_f], [run_amp], [walk_amp], ...)
    """
    assert pos_amp_all is None or len(pos_amp_all) == len(pos_f_all), \
        "pos_amp_all must be same length as pos_f_all"
    assert neg_amp_all is None or len(neg_amp_all) == len(neg_f_all), \
        "neg_amp_all must be same length as neg_f_all"

    # Generate the labels, ~1/2 from + and 1/2 from - classes
    labels = np.random.randint(2, size=n)

    # Multi-dimensional data
    data = []

    # For each channel
    for channel in range(len(pos_f_all)):
        pos_f = np.array(pos_f_all[channel], dtype=np.float32)
        neg_f = np.array(neg_f_all[channel], dtype=np.float32)

        assert isinstance(freq_noise, list), \
            "freq_noise should be list [noise ch1, noise ch2, ...]"
        assert isinstance(amp_noise, list), \
            "amp_noise should be list [noise ch1, noise ch2, ...]"
        freq_noise_channel = freq_noise[channel]
        amp_noise_channel = amp_noise[channel]

        # Check that we don't accidentally expect multiple output channels for
        # instance but only create one
        assert len(freq_noise) == len(pos_f_all), \
            "more noise channels than frequency channels"
        assert len(amp_noise) == len(pos_f_all), \
            "more noise channels than frequency channels"

        if pos_amp_all is not None:
            pos_amp = np.array(pos_amp_all[channel], dtype=np.float32)
        else:
            pos_amp = np.array([1.0]*len(pos_f_all), dtype=np.float32)

        if neg_amp_all is not None:
            neg_amp = np.array(neg_amp_all[channel], dtype=np.float32)
        else:
            neg_amp = np.array([1.0]*len(neg_f_all), dtype=np.float32)

        assert pos_amp is None or len(pos_amp) == len(pos_f), \
            "pos_amp must be same length as pos_f"
        assert neg_amp is None or len(neg_amp) == len(neg_f), \
            "neg_amp must be same length as neg_f"

        # Match sizes by zero padding, otherwise we can't convert to single matrix
        if len(pos_f) > len(neg_f):
            padding = len(pos_f) - len(neg_f)
            neg_f = np.pad(neg_f, (0, padding), 'constant', constant_values=(0.0, 0.0))
            neg_amp = np.pad(neg_amp, (0, padding), 'constant', constant_values=(0.0, 0.0))
        elif len(neg_f) > len(pos_f):
            padding = len(neg_f) - len(pos_f)
            pos_f = np.pad(pos_f, (0, padding), 'constant', constant_values=(0.0, 0.0))
            pos_amp = np.pad(pos_amp, (0, padding), 'constant', constant_values=(0.0, 0.0))

        # Get approximately the pos_f/neg_f frequencies for each
        freqs = []
        amps = []

        for label in labels:
            f = neg_f if label == 0 else pos_f
            amp = neg_amp if label == 0 else pos_amp
            freqs.append(f)
            amps.append(amp)

        freqs = np.array(freqs, dtype=np.float32)
        amps = np.array(amps, dtype=np.float32)
        amps[np.isnan(amps)] = 1.0  # None is NaN, and so just set to 1.0 amplitude

        # Generate time series data
        x, y = sine(b=b, f=freqs, amps=amps, maxt=duration, length=duration*sample_freq,
            freq_noise=freq_noise_channel, phase_shift=phase_shift)

        if amp_noise is not None:
            y += np.random.normal(0.0, amp_noise_channel, (y.shape[0], n))

        if display:
            display_xy(x, y)

        data.append(y)

    # Transpose from [features, time_steps, examples] to
    # [examples, time_steps, features]
    data = np.array(data, dtype=np.float32).T

    # Make labels 1-indexed
    labels += 1

    return data, labels


def rotate2d(x, degrees):
    """
    Rotate the 2D data in x a certain number of degrees (clockwise, if each
    point in x is (x,y))

    If x is a single point (i.e. x is something like (x,y)) then it's only
    rotates that point. However, more useful is if x is a time series, where the
    values to be rotated (i.e. the feature dimension) is last. For example, pass
    x with shape: [examples, time_steps, num_features] where num_features = 2
    (since this is a 2D rotation matrix).

    Note: if you want counterclockwise, do a left-multiply instead of a
    right-multiply

    See:
    https://en.wikipedia.org/wiki/Rotation_matrix
    https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/
    """
    theta = np.radians(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    return np.dot(x, rotation_matrix)


def rotate2d_data(data, labels, degrees):
    """ rotate2d but only rotates data and directly passes through labels """
    return rotate2d(data, degrees), labels


def save_data_file(values, labels, filename):
    """
    To be compatible with UCR dataset format with 1D data (univariate), commas
    separate label first then all the data with one example on each line.
    However, to support multivariate data, features for each time step are
    delimitated by semicolons.

    Example:
      univariate: label,timestep1,timestep2,timestep3,...
      multivariate: label,ts1f1;ts1f2;ts1f3,fs2f1;fs2f2;ts2f3,...
    """
    with open(filename, "w") as f:
        for i, x in enumerate(values):
            y = labels[i]
            s = str(y) + ","

            # If only one feature, we don't have the extra dimension
            if len(x.shape) == 1:
                s += ",".join([str(v) for v in x])
            elif len(x.shape) == 2:
                for j, time_step in enumerate(x):
                    s += ";".join([str(v) for v in time_step])

                    if j != len(x) - 1:
                        s += ","
            else:
                raise NotImplementedError(
                    "only support shapes [examples, time_steps]"
                    " or [examples, time_steps, features]")

            f.write(s+"\n")


def save_data(func, dataset_name, display=False):
    """ Use func to create examples that are saved to ..._TRAIN and ..._TEST """
    print(dataset_name)
    save_data_file(*func(50000, False), 'datasets/synthetic/'+dataset_name+'_TRAIN')
    save_data_file(*func(5000, display), 'datasets/synthetic/'+dataset_name+'_TEST')


if __name__ == '__main__':
    if not os.path.exists('datasets/synthetic'):
        os.makedirs('datasets/synthetic')

    # Whether to display
    dsp = False

    # # For reproducibility
    # np.random.seed(0)
    # # No noise
    # save_data(lambda x, dsp: generate_positive_slope_data(x, display=dsp), 'positive_slope', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, display=dsp), 'positive_sine', dsp)
    # # Noisy
    # save_data(lambda x, dsp: generate_positive_slope_data(x, add_noise=True, display=dsp), 'positive_slope_noise', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, add_noise=True, display=dsp), 'positive_sine_noise', dsp)
    # # No noise - but different y-intercept
    # save_data(lambda x, dsp: generate_positive_slope_data(x, bmin=20.0, bmax=30.0, display=dsp), 'positive_slope_low', dsp)
    # save_data(lambda x, dsp: generate_positive_sine_data(x, bmin=20.0, bmax=30.0, display=dsp), 'positive_sine_low', dsp)

    # # Frequency
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, display=dsp), 'freq_low', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, display=dsp), 'freq_high', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, amp_noise=True, display=dsp), 'freq_low_amp_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, amp_noise=True, display=dsp), 'freq_high_amp_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, freq_noise=True, display=dsp), 'freq_low_freq_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, freq_noise=True, display=dsp), 'freq_high_freq_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=0.1, fmax=0.5, amp_noise=True, freq_noise=True, display=dsp), 'freq_low_freqamp_noise', dsp)
    # save_data(lambda x, dsp: generate_freq(x, fmin=1.0, fmax=5.0, amp_noise=True, freq_noise=True, display=dsp), 'freq_high_freqamp_noise', dsp)

    # # Multiple frequencies
    # # TODO maybe remove/decrease frequency/amplitude noise and reduce domain shifts
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, [2, 6, 8], [7, 9, 11], display=dsp), 'freqshift_low', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, [8, 12, 14], [13, 15, 17], display=dsp), 'freqshift_high', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, [2, 6, 8], [7, 9, 11], display=dsp), 'freqscale_low', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, [4, 12, 16], [14, 18, 22], display=dsp), 'freqscale_high', dsp)

    # #
    # # Classification problem:
    # # Walking (negative) vs. running (positive)
    # #
    # run_f = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    # run_amp = np.array([0.75, 0.56, 0.38, 0.19, 0.08, 0.04], dtype=np.float32)
    # walk_f = np.array([1, 2, 4], dtype=np.float32)
    # walk_amp = np.array([0.5, 0.25, 0.06], dtype=np.float32)

    # # Since we want 1D data and I modified generate_multi_freq to support
    # # multi-dimensional data, but use numpy arrays so we can still do
    # # addition/multiplication on these
    # run_f = np.expand_dims(run_f, axis=0)
    # run_amp = np.expand_dims(run_amp, axis=0)
    # walk_f = np.expand_dims(walk_f, axis=0)
    # walk_amp = np.expand_dims(walk_amp, axis=0)

    # # Frequency shift
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None), 'freqshift_a', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+0.00, walk_f+0.00, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b0', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+3.80, walk_f+3.80, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b1', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+7.60, walk_f+7.60, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b2', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+11.4, walk_f+11.4, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b3', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+15.2, walk_f+15.2, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b4', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+19.0, walk_f+19.0, run_amp, walk_amp, dsp, None, None, None), 'freqshift_b5', dsp)

    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_a', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+0.00, walk_f+0.00, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b0', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+3.80, walk_f+3.80, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b1', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+7.60, walk_f+7.60, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b2', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+11.4, walk_f+11.4, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b3', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+15.2, walk_f+15.2, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b4', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+19.0, walk_f+19.0, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b5', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+22.8, walk_f+22.8, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b6', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+26.6, walk_f+26.6, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b7', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+30.4, walk_f+30.4, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b8', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+34.2, walk_f+34.2, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b9', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+38.0, walk_f+38.0, run_amp, walk_amp, dsp, None, None), 'freqshift_phase_b10', dsp)

    # # Frequency scale (shift looks identical, but scaled horizontally)
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None, None), 'freqscale_a', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.000, walk_f*1.000, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b0', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.633, walk_f*1.633, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b1', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.266, walk_f*2.266, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b2', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.900, walk_f*2.900, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b3', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*3.533, walk_f*3.533, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b4', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*4.166, walk_f*4.166, run_amp, walk_amp, dsp, None, None, None), 'freqscale_b5', dsp)

    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_a', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.000, walk_f*1.000, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b0', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.633, walk_f*1.633, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b1', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.266, walk_f*2.266, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b2', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.900, walk_f*2.900, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b3', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*3.533, walk_f*3.533, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b4', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*4.166, walk_f*4.166, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b5', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*4.800, walk_f*4.800, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b6', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*5.433, walk_f*5.433, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b7', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*6.067, walk_f*6.067, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b8', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*6.700, walk_f*6.700, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b9', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*7.333, walk_f*7.333, run_amp, walk_amp, dsp, None, None), 'freqscale_phase_b10', dsp)

    # # Frequency scale and shift
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_a', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.000+0.00, walk_f*1.000+0.00, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b0', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*1.633+3.80, walk_f*1.633+3.80, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b1', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.266+7.60, walk_f*2.266+7.60, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b2', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*2.900+11.4, walk_f*2.900+11.4, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b3', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*3.533+15.2, walk_f*3.533+15.2, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b4', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*4.166+19.0, walk_f*4.166+19.0, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b5', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*4.800+22.8, walk_f*4.800+22.8, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b6', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*5.433+26.6, walk_f*5.433+26.6, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b7', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*6.067+30.4, walk_f*6.067+30.4, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b8', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*6.700+34.2, walk_f*6.700+34.2, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b9', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f*7.333+38.0, walk_f*7.333+38.0, run_amp, walk_amp, dsp, None, None), 'freqscaleshift_phase_b10', dsp)

    # #
    # # 2D classification problem:
    # # Walking (negative) vs. running (positive)
    # #
    # # Split first half / second half of frequencies into x vs. y rather than
    # # every other one so that the two channels look quite different. Hopefully
    # # this makes adaptation more useful.
    # #
    # run_f_x = np.array([1, 2, 3], dtype=np.float32)
    # run_amp_x = np.array([0.75, 0.56, 0.38], dtype=np.float32)
    # run_f_y = np.array([4, 5, 6], dtype=np.float32)
    # run_amp_y = np.array([0.19, 0.08, 0.04], dtype=np.float32)

    # walk_f_x = np.array([1, 2], dtype=np.float32)
    # walk_amp_x = np.array([0.5, 0.25], dtype=np.float32)
    # walk_f_y = np.array([4, 0], dtype=np.float32)  # match shape
    # walk_amp_y = np.array([0.06, 0], dtype=np.float32)

    # # Keep as numpy arrays since otherwise we can't add/multiply for shifts
    # run_f = np.array([run_f_x, run_f_y], dtype=np.float32)
    # run_amp = np.array([run_amp_x, run_amp_y], dtype=np.float32)
    # walk_f = np.array([walk_f_x, walk_f_y], dtype=np.float32)
    # walk_amp = np.array([walk_amp_x, walk_amp_y], dtype=np.float32)

    # # Linear transform (random/fixed), rotating from 0 degrees to 180 degrees (e.g. 2D accelerometer)
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'rotate2_phase_a', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 0), 'rotate2_phase_b0', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 18), 'rotate2_phase_b1', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 36), 'rotate2_phase_b2', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 54), 'rotate2_phase_b3', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 72), 'rotate2_phase_b4', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 90), 'rotate2_phase_b5', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 108), 'rotate2_phase_b6', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 126), 'rotate2_phase_b7', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 144), 'rotate2_phase_b8', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 162), 'rotate2_phase_b9', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 180), 'rotate2_phase_b10', dsp)

    # # np.random.seed(0)
    # # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 'rotate2_noise_a', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 0), 'rotate2_noise_b0', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 18), 'rotate2_noise_b1', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 36), 'rotate2_noise_b2', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 54), 'rotate2_noise_b3', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 72), 'rotate2_noise_b4', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 90), 'rotate2_noise_b5', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 108), 'rotate2_noise_b6', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 126), 'rotate2_noise_b7', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 144), 'rotate2_noise_b8', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 162), 'rotate2_noise_b9', dsp)
    # # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp), 180), 'rotate2_noise_b10', dsp)

    # # Shift and rotate
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'freqshiftrotate_phase_a', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+0.00, walk_f+0.00, run_amp, walk_amp, dsp, None, None), 0), 'freqshiftrotate_phase_b0', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+3.80, walk_f+3.80, run_amp, walk_amp, dsp, None, None), 18), 'freqshiftrotate_phase_b1', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+7.60, walk_f+7.60, run_amp, walk_amp, dsp, None, None), 36), 'freqshiftrotate_phase_b2', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+11.4, walk_f+11.4, run_amp, walk_amp, dsp, None, None), 54), 'freqshiftrotate_phase_b3', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+15.2, walk_f+15.2, run_amp, walk_amp, dsp, None, None), 72), 'freqshiftrotate_phase_b4', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+19.0, walk_f+19.0, run_amp, walk_amp, dsp, None, None), 90), 'freqshiftrotate_phase_b5', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+22.8, walk_f+22.8, run_amp, walk_amp, dsp, None, None), 108), 'freqshiftrotate_phase_b6', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+26.6, walk_f+26.6, run_amp, walk_amp, dsp, None, None), 126), 'freqshiftrotate_phase_b7', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+30.4, walk_f+30.4, run_amp, walk_amp, dsp, None, None), 144), 'freqshiftrotate_phase_b8', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+34.2, walk_f+34.2, run_amp, walk_amp, dsp, None, None), 162), 'freqshiftrotate_phase_b9', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f+38.0, walk_f+38.0, run_amp, walk_amp, dsp, None, None), 180), 'freqshiftrotate_phase_b10', dsp)

    # # Scale and rotate
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'freqscalerotate_phase_a', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*1.000, walk_f*1.000, run_amp, walk_amp, dsp, None, None), 0), 'freqscalerotate_phase_b0', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*1.633, walk_f*1.633, run_amp, walk_amp, dsp, None, None), 18), 'freqscalerotate_phase_b1', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*2.266, walk_f*2.266, run_amp, walk_amp, dsp, None, None), 36), 'freqscalerotate_phase_b2', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*2.900, walk_f*2.900, run_amp, walk_amp, dsp, None, None), 54), 'freqscalerotate_phase_b3', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*3.533, walk_f*3.533, run_amp, walk_amp, dsp, None, None), 72), 'freqscalerotate_phase_b4', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*4.166, walk_f*4.166, run_amp, walk_amp, dsp, None, None), 90), 'freqscalerotate_phase_b5', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*4.800, walk_f*4.800, run_amp, walk_amp, dsp, None, None), 108), 'freqscalerotate_phase_b6', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*5.433, walk_f*5.433, run_amp, walk_amp, dsp, None, None), 126), 'freqscalerotate_phase_b7', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*6.067, walk_f*6.067, run_amp, walk_amp, dsp, None, None), 144), 'freqscalerotate_phase_b8', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*6.700, walk_f*6.700, run_amp, walk_amp, dsp, None, None), 162), 'freqscalerotate_phase_b9', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*7.333, walk_f*7.333, run_amp, walk_amp, dsp, None, None), 180), 'freqscalerotate_phase_b10', dsp)

    # # Scale, shift, and rotate
    # np.random.seed(0)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, None, None), 'freqscaleshiftrotate_phase_a', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*1.000+0.00, walk_f*1.000+0.00, run_amp, walk_amp, dsp, None, None), 0), 'freqscaleshiftrotate_phase_b0', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*1.633+3.80, walk_f*1.633+3.80, run_amp, walk_amp, dsp, None, None), 18), 'freqscaleshiftrotate_phase_b1', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*2.266+7.60, walk_f*2.266+7.60, run_amp, walk_amp, dsp, None, None), 36), 'freqscaleshiftrotate_phase_b2', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*2.900+11.4, walk_f*2.900+11.4, run_amp, walk_amp, dsp, None, None), 54), 'freqscaleshiftrotate_phase_b3', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*3.533+15.2, walk_f*3.533+15.2, run_amp, walk_amp, dsp, None, None), 72), 'freqscaleshiftrotate_phase_b4', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*4.166+19.0, walk_f*4.166+19.0, run_amp, walk_amp, dsp, None, None), 90), 'freqscaleshiftrotate_phase_b5', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*4.800+22.8, walk_f*4.800+22.8, run_amp, walk_amp, dsp, None, None), 108), 'freqscaleshiftrotate_phase_b6', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*5.433+26.6, walk_f*5.433+26.6, run_amp, walk_amp, dsp, None, None), 126), 'freqscaleshiftrotate_phase_b7', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*6.067+30.4, walk_f*6.067+30.4, run_amp, walk_amp, dsp, None, None), 144), 'freqscaleshiftrotate_phase_b8', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*6.700+34.2, walk_f*6.700+34.2, run_amp, walk_amp, dsp, None, None), 162), 'freqscaleshiftrotate_phase_b9', dsp)
    # save_data(lambda x, dsp: rotate2d_data(*generate_multi_freq(x, run_f*7.333+38.0, walk_f*7.333+38.0, run_amp, walk_amp, dsp, None, None), 180), 'freqscaleshiftrotate_phase_b10', dsp)

    #
    # Classification problem:
    # Walking (negative) vs. running (positive)
    #
    run_f = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    run_amp = np.array([0.75, 0.56, 0.38, 0.19, 0.08, 0.04], dtype=np.float32)
    walk_f = np.array([1, 2, 4], dtype=np.float32)
    walk_amp = np.array([0.5, 0.25, 0.06], dtype=np.float32)

    # Since we want 1D data and I modified generate_multi_freq to support
    # multi-dimensional data, but use numpy arrays so we can still do
    # addition/multiplication on these
    run_f = np.expand_dims(run_f, axis=0)
    run_amp = np.expand_dims(run_amp, axis=0)
    walk_f = np.expand_dims(walk_f, axis=0)
    walk_amp = np.expand_dims(walk_amp, axis=0)

    # Tile to make two channels. The data (signal) will be the same but the
    # noise will be different.
    run_f = np.tile(run_f, (2, 1))
    run_amp = np.tile(run_amp, (2, 1))
    walk_f = np.tile(walk_f, (2, 1))
    walk_amp = np.tile(walk_amp, (2, 1))

    # Original data but with "data augmentation" rather than a second modality
    np.random.seed(0)
    amp_noise = [1.0]*2
    freq_noise = [1.0]*2
    save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_aa', dsp)

    # Noise for modality 1 then 2
    # syn
    # amp_noise = [5.0, 0.5]
    # freq_noise = [0.5, 5.0]
    # syn2
    amp_noise = [1.0, 0.1]
    freq_noise = [0.1, 1.0]

    # Frequency shift
    np.random.seed(0)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f, walk_f, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_a', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+0.00, walk_f+0.00, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b0', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+3.80, walk_f+3.80, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b1', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+7.60, walk_f+7.60, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b2', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+11.4, walk_f+11.4, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b3', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+15.2, walk_f+15.2, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b4', dsp)
    save_data(lambda x, dsp: generate_multi_freq(x, run_f+19.0, walk_f+19.0, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b5', dsp)
    # below these probably alias since >1/2f (max f is 6 Hz for run, 19+6 = 25 = 1/2 sampling rate of 50 Hz)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+22.8, walk_f+22.8, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b6', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+26.6, walk_f+26.6, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b7', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+30.4, walk_f+30.4, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b8', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+34.2, walk_f+34.2, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b9', dsp)
    # save_data(lambda x, dsp: generate_multi_freq(x, run_f+38.0, walk_f+38.0, run_amp, walk_amp, dsp, amp_noise, freq_noise), 'freqshift_b10', dsp)

    #
    # Multi-dimensional
    #
    np.random.seed(0)

    # Number of dimensions
    k = 6

    # Randomly generate the frequency/amplitudes of some signals for each class
    # and dimension
    def _gen_freq_amp(k,
            num_components_low=1, num_components_high=5,
            freq_low=1, freq_high=25,
            amp_low=0, amp_high=1):
        freqs = []
        amps = []

        for dim in range(k):  # k dimensions
            freq = []
            amp = []

            # Randomly pick number of frequency components
            n = np.random.randint(num_components_low, num_components_high)

            # Randomly pick that many frequency/amplitude
            for i in range(n):
                freq.append(np.random.uniform(freq_low, freq_high))
                amp.append(np.random.uniform(amp_low, amp_high))

            freqs.append(freq)
            amps.append(amp)

        return freqs, amps

    # Randomly adjust each frequency by some amount
    def _freq_noise(freqs, amps, noise_amount):
        return [
            [
                freq + np.random.normal(scale=noise_amount)
                for freq in dim_freqs
            ]
            for dim_freqs in freqs
        ], amps

    # Source domain
    freq_pos, amp_pos = _gen_freq_amp(k)
    freq_neg, amp_neg = _gen_freq_amp(k)

    amp_noise = [0]*k
    freq_noise = [0]*k

    save_data(lambda x, dsp: generate_multi_freq(x, freq_pos, freq_neg, amp_pos, amp_neg, dsp, amp_noise, freq_noise), 'multidim_a', dsp)
    # print("Positive:", freq_pos, amp_pos)
    # print("Negative:", freq_neg, amp_neg)

    # Target domains
    noise_amounts = {
        "b1": 1,
        "b2": 2,
        "b3": 3,
        "b4": 4,
        "b5": 5,
    }

    for domain, noise_amount in noise_amounts.items():
        freq_pos, amp_pos = _freq_noise(freq_pos, amp_pos, noise_amount)
        freq_neg, amp_neg = _freq_noise(freq_neg, amp_neg, noise_amount)

        # print("Positive:", freq_pos, amp_pos)
        # print("Negative:", freq_neg, amp_neg)
        # break

        save_data(lambda x, dsp: generate_multi_freq(x, freq_pos, freq_neg, amp_pos, amp_neg, dsp, amp_noise, freq_noise), 'multidim_' + domain, dsp)
