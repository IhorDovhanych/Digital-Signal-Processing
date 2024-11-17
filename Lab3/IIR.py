import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

srate = 1024
nyquist = srate / 2

frange = [20, 45]

order = 2

fkernB, fkernA = butter(order, np.array(frange) / nyquist, btype='band')

plt.figure(1)
plt.subplot(221)
plt.plot(fkernB * 1e5, 'ks-', linewidth=2, markersize=3, markerfacecolor='w', label='B')
plt.plot(fkernA, 'rs-', linewidth=2, markersize=3, markerfacecolor='w', label='A')
plt.xlabel('Time points')
plt.ylabel('Filter coeffs.')
plt.title('Time-domain filter coefficients')
plt.legend()

filtpow = np.abs(np.fft.fft(fkernB))**2
hz = np.linspace(0, srate / 2, len(fkernB) // 2 + 1)

plt.subplot(222)
plt.stem(hz, filtpow[:len(hz)], linefmt='ks-', markerfmt='ko', basefmt=" ")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power spectrum of filter coeffs.')

impres = np.concatenate([np.zeros(500), [1], np.zeros(500)])
fimp = lfilter(fkernB, fkernA, impres)

fimpX = np.abs(np.fft.fft(fimp))**2
hz = np.linspace(0, nyquist, len(impres) // 2 + 1)

plt.subplot(223)
plt.plot(impres, 'k', linewidth=2)
plt.plot(fimp, 'r', linewidth=2)
plt.xlim([1, len(impres)])
plt.ylim([-1 * 0.06, 1 * 0.06])
plt.legend(['Impulse', 'Filtered'])
plt.xlabel('Time points (a.u.)')
plt.title('Filtering an impulse')

plt.subplot(224)
plt.plot(hz, 10 * np.log10(fimpX[:len(hz)]), 'ks-', linewidth=2, markerfacecolor='w', markersize=3)
plt.xlim([0, 100])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (log)')
plt.title('Frequency response of filter (Butterworth)')

plt.tight_layout()
plt.show()
