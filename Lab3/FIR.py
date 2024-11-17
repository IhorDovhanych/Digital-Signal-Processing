import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, freqz

srate = 1024
nyquist = srate / 2
frange = [10, 65]
order = 1200

filtkern = firwin(order + 1, np.array(frange) / nyquist)

filtpow = np.abs(np.fft.fft(filtkern)) ** 2

hz = np.linspace(0, srate / 2, len(filtkern) // 2 + 1)
filtpow = filtpow[:len(hz)]

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(filtkern, linewidth=2)
plt.xlabel('Time points')
plt.title('Filter kernel (fir1)')
plt.xlim([0, 1000])
plt.ylim([-0.06, 0.06])

impres = np.concatenate([np.zeros(500), [1], np.zeros(500)])

fimp = lfilter(filtkern, 1, impres)

plt.subplot(2, 2, 2)
plt.plot(impres, 'k', linewidth=2)
plt.plot(fimp, 'r', linewidth=2)
plt.xlim([0, len(impres)])
plt.ylim([-0.06, 0.06])
plt.legend(['Impulse', 'Filtered'])
plt.xlabel('Time points (a.u.)')
plt.title('Filtering an impulse')

plt.subplot(2, 2, 3)
plt.plot(hz, filtpow, 'ks-', linewidth=2, markersize=2, markerfacecolor='w')

plt.plot([0, frange[0], *frange, frange[1], nyquist], [0, 0, 1, 1, 0, 0], 'ro-', linewidth=2, markersize=2, markerfacecolor='w')

plt.plot([frange[0], frange[0]], plt.gca().get_ylim(), 'k:', markersize=2)

plt.xlim([0, frange[0] * 4])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain')
plt.legend(['Actual', 'Ideal'])
plt.title('Frequency response of filter (fir1)')
plt.xlim([0, 1000])
plt.ylim([-0.06, 0.06])

plt.subplot(2, 2, 4)
plt.plot(hz, 10 * np.log10(filtpow), 'ks-', linewidth=2, markersize=2, markerfacecolor='w')
plt.plot([frange[0], frange[0]], plt.gca().get_ylim(), 'k:')

plt.xlim([0, frange[0] * 4])
plt.ylim([-80, 2])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Filter gain (dB)')
plt.title('Frequency response of filter (fir1)')
plt.xlim([0, 1000])
plt.ylim([-0.06, 0.06])

plt.tight_layout()
plt.show()
