import numpy as np
import math

################################################
# Audio processing constants and functions
################################################
fps = 22000            # audio frames per second
cps = 10               # cuts per second
fpc = int(fps / cps)   # frames per cut
beep_freq_tolerance = 50

def goertzel(samples, sample_rate, *freqs):
    """
    Implementation of the Goertzel algorithm, useful for calculating individual
    terms of a discrete Fourier transform.

    `samples` is a windowed one-dimensional signal originally sampled at `sample_rate`.

    The function returns 2 arrays, one containing the actual frequencies calculated,
    the second the coefficients `(real part, imag part, power)` for each of those frequencies.
    For simple spectral analysis, the power is usually enough.

    Example of usage :

        # calculating frequencies in ranges [400, 500] and [1000, 1100]
        # of a windowed signal sampled at 44100 Hz

        freqs, results = goertzel(some_samples, 44100, (400, 500), (1000, 1100))
    """
    window_size = len(samples)
    f_step = sample_rate / float(window_size)
    f_step_normalized = 1.0 / window_size

    # Calculate all the DFT bins we have to compute to include frequencies
    # in `freqs`.
    bins = set()
    for f_range in freqs:
        f_start, f_end = f_range
        k_start = int(math.floor(f_start / f_step))
        k_end = int(math.ceil(f_end / f_step))

        if k_end > window_size - 1: raise ValueError('frequency out of range %s' % k_end)
        bins = bins.union(range(k_start, k_end))

    # For all the bins, calculate the DFT term
    n_range = range(0, window_size)
    freqs = []
    results = []
    for k in bins:

        # Bin frequency and coefficients for the computation
        f = k * f_step_normalized
        w_real = 2.0 * math.cos(2.0 * math.pi * f)
        w_imag = math.sin(2.0 * math.pi * f)

        # Doing the calculation on the whole sample
        d1, d2 = 0.0, 0.0
        for n in n_range:
            y  = samples[n] + w_real * d1 - d2
            d2, d1 = d1, y

        # Storing results `(real part, imag part, power)`
        results.append((
            0.5 * w_real * d1 - d2, w_imag * d1,
            d2**2 + d1**2 - w_real * d1 * d2)
        )
        freqs.append(f * sample_rate)
    return freqs, results

################################################
# Detect runs start / stop times by beeps
# Both run start and run stop beeps are of the same frequency
################################################
def detect_beeps( audioclip, beep_freq=500, beep_power_threshold=50):
    
    # cut audio into subsecond cuts
    cut = lambda i: audioclip.subclip(i/cps,(i+1)/cps).to_soundarray(fps=22000, quantize=False)
    
    samples = lambda array: array[:fpc][:, 0]   # take one column of audio signal
    max_pwr = lambda array, freq: max([r[2] for r in goertzel( samples(array), fps, freq)[1]])

    num_cuts = int(audioclip.duration*cps - 1)

    ################################################
    # Detect beeps
    # Same note designates BOTH an experiment start and end
    ################################################
    beep_pwrs  = [max_pwr(cut(i), (beep_freq - beep_freq_tolerance, beep_freq + beep_freq_tolerance)) for i in range(num_cuts)]
    
    beep_starts = [i / cps                    # time in seconds             
                   for i, (cur, next) 
                   in enumerate(zip(beep_pwrs[:-1], beep_pwrs[1:])) 
                   if cur < beep_power_threshold and next > beep_power_threshold or i==0 and cur > beep_power_threshold]

    # return list of tuples of (experiment_start, experiment_end)
    return zip( beep_starts[0::2], beep_starts[1::2])