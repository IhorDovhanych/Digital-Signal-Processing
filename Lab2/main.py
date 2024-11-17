import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import spectrogram
from scipy.signal.windows import hann
import sounddevice as sd

# Зчитуємо сигнал і частоту дискретизації
Fs, sig = wav.read('./XC403881.wav')
# Fs, sig = wav.read('./XC307466.wav')

# Вибираємо канал (лівий/правий)
if len(sig.shape) > 1:
    sig = sig[:, 0]

# Відтворюємо аудіофайл
print("Програвання аудіофайлу...")
sd.play(sig, Fs)
sd.wait()  # Чекаємо завершення відтворення

# Довжина сигналу
sig_len = len(sig)

# Визначаємо довжину в секундах
sec = sig_len / Fs

# Створюємо перший графік (спектрограма через scipy)
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)  # Створюємо область для першого графіка
f, t, Sxx = spectrogram(sig, Fs, window='hann', nperseg=1000, noverlap=500, scaling='spectrum')
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Scipy Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='dB')

# Довжина вікна
wndw_sz = 500
# Величина перекриття
overlap_sz = 250
# Точки в сигналі де буде вікно Ганна
wndw_pts = np.arange(0, sig_len - wndw_sz, wndw_sz - overlap_sz)

# Масив в якому будуть зберігатись перетворення Фур'є вікон
spgram = np.zeros((wndw_sz, len(wndw_pts)), dtype=np.complex64)
spgram_sec = np.zeros((wndw_sz, len(wndw_pts)), dtype=np.complex64)
# Функція Ганна по величині вікна
from scipy.signal.windows import hamming
hamm_func = hamming(wndw_sz)

hann_func = hann(wndw_sz)
# Проходимо по сигналу і робимо перетворення Фур'є в кожному вікні і зберігаємо в матрицю
for i, pt in enumerate(wndw_pts):
    windowed_signal = sig[pt:pt + wndw_sz] * hann_func
    spgram[:, i] = np.fft.fft(windowed_signal)
for i, pt in enumerate(wndw_pts):
    windowed_signal = sig[pt:pt + wndw_sz] * hamm_func
    spgram_sec[:, i] = np.fft.fft(windowed_signal)
# Знаходимо частоту Найквіста для коректного виводу
max_freq = wndw_sz // 2

# Нормалізуємо коефіцієнти Фур'є
spgram[:, 1:] = spgram[:, 1:] * 2
spgram_sec[:, 1:] = spgram[:, 1:] * 2

# Знаходимо амплітуди для виводу їх на екран як зображення
img = np.abs(spgram[:max_freq, :])
nyquist_f = Fs / 2

# Створюємо другий графік (спектрограма через кастомний підхід)
plt.subplot(1, 3, 2)  # Створюємо область для другого графіка
plt.imshow(img * 15, aspect='auto', extent=[0, sec, 0, nyquist_f], origin='lower')
plt.title('Custom Spectrogram')
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Magnitude')

# Знаходимо амплітуди для виводу їх на екран як зображення
img_sec = np.abs(spgram[:max_freq, :])
plt.subplot(1, 3, 3)  # Створюємо область для другого графіка
plt.imshow(img_sec * 15, aspect='auto', extent=[0, sec, 0, nyquist_f], origin='lower')
plt.title('Custom Spectrogram')
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Magnitude')
# Виводимо обидва графіка одночасно
plt.tight_layout()  # Підганяємо макет для уникнення перекриття елементів
plt.show()

# Вивести частоту дискретизації та довжину сигналу
print(f"Частота дискретизації: {Fs} Hz")
print(f"Довжина сигналу: {sig_len} зразків")
# Вивести тривалість сигналу в секундах
print(f"Тривалість сигналу: {sec:.2f} секунд")
# Вивести кілька значень частот і часу для спектрограми
print(f"Перша частота у спектрограмі: {f[0]} Hz")
print(f"Остання частота у спектрограмі: {f[-1]} Hz")
print(f"Перший момент часу у спектрограмі: {t[0]} с")
print(f"Останній момент часу у спектрограмі: {t[-1]} с")
# Вивести розмір матриці спектрограми
print(f"Розмір спектрограми: {Sxx.shape}")
# Вивести кілька значень перетворення Фур'є для першого вікна
print(f"Перше значення перетворення Фур'є для першого вікна: {spgram[0, 0]}")
print(f"Останнє значення перетворення Фур'є для першого вікна: {spgram[-1, 0]}")
# Вивести мінімальну і максимальну амплітуду для кастомної спектрограми
print(f"Мінімальна амплітуда: {np.min(img)}")
print(f"Максимальна амплітуда: {np.max(img)}")
# Вивести частоту Найквіста
print(f"Частота Найквіста: {nyquist_f} Hz")
