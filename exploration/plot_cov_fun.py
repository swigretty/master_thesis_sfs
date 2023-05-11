import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from statsmodels.tsa.arima_process import ArmaProcess

if __name__ == "__main__":
    # Define AR(1) process parameters
    a = 0.5
    ar = np.array([1, a])
    variance = 1
    n_samples = 500
    x = ArmaProcess(ar, np.array([1])).generate_sample(500, scale=variance)
    # Define frequency range
    freq = np.linspace(0, 1, n_samples)

    # Calculate PSD using signal.periodogram
    freq, psd = signal.periodogram(x, fs=1, window='boxcar', return_onesided=False)

    # Normalize PSD by variance
    psd /= variance

    # Plot PSD
    plt.plot(freq, psd, linewidth=2)
    plt.xlabel('Frequency')
    plt.ylabel('Power spectral density')
    plt.title('Power spectral density of AR(1) process')
    plt.show()

    plt.plot(range(len(x)), x)
    print(psd)

