import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firls, freqz
from scipy.fft import fft

# Параметри
srate = 1024
nyquist = srate / 2
frange = [20, 45]
transwLenL = 0.05
transwLenR = 0.05
order = 501

shape = [0, 0, 1, 1, 0, 0]
frex = [0, frange[0] - frange[0] * transwLenL, *frange, frange[1] + frange[1] * transwLenR, nyquist]
frex = np.array([f / nyquist for f in frex])

filtkern = firls(order, frex, shape)

filtpow = np.abs(fft(filtkern))**2
hz = np.linspace(0, srate / 2, len(filtkern) // 2 + 1)
filtpow = filtpow[:len(hz)]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(filtkern, linewidth=2)
axs[0, 0].set_xlabel('Time points')
axs[0, 0].set_title('Filter kernel (firls)')

impres = np.concatenate([np.zeros(500), [1], np.zeros(500)])
fimp = np.convolve(impres, filtkern, mode='same')

axs[0, 1].plot(impres, 'k', linewidth=2)
axs[0, 1].plot(fimp, 'r', linewidth=2)
axs[0, 1].set_xlim([0, len(impres)])
axs[0, 1].set_ylim([-1 * 0.06, 1 * 0.06])  # Масштаб
axs[0, 1].legend(['Impulse', 'Filtered'])
axs[0, 1].set_xlabel('Time points (a.u.)')
axs[0, 1].set_title('Filtering an impulse')

axs[1, 0].plot(frex * nyquist, shape, 'ro-', linewidth=2, markersize=3, markerfacecolor='w')
axs[1, 0].plot(hz, filtpow, 'ks-', linewidth=2, markersize=3, markerfacecolor='w')
axs[1, 0].set_xlim([0, frange[0] * 4])
axs[1, 0].set_xlabel('Frequency (Hz)')
axs[1, 0].set_ylabel('Filter gain')
axs[1, 0].legend(['Actual', 'Ideal'])
axs[1, 0].set_title('Frequency response of filter (firls)')

axs[1, 1].plot(hz, 10 * np.log10(filtpow), 'ks-', linewidth=2, markersize=3, markerfacecolor='w')
axs[1, 1].plot([frange[0], frange[0]], axs[1, 1].get_ylim(), 'k:')  # Вертикальна лінія на границі фільтрації
axs[1, 1].set_xlim([0, frange[0] * 4])
axs[1, 1].set_ylim([-50, 2])
axs[1, 1].set_xlabel('Frequency (Hz)')
axs[1, 1].set_ylabel('Filter gain (dB)')
axs[1, 1].set_title('Frequency response of filter (firls)')

plt.tight_layout()
plt.show()
