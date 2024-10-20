import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import pinv

def create_signal_with_noise(srate, sec, signal_type, noise_amplitude, frequency=1, amplitude=None, phase=None, DC=None, poles=10, lintrend=False):
    if amplitude is None:
        amplitude = np.ones(len(np.atleast_1d(frequency)))
    if phase is None:
        phase = np.zeros(len(np.atleast_1d(frequency)))
    if DC is None:
        DC = np.zeros(len(np.atleast_1d(frequency)))

    xtime = np.arange(0, sec, 1/srate)

    if signal_type == "sine":
        signal = np.zeros(len(xtime))
        for i, freq in enumerate(np.atleast_1d(frequency)):
            signal += amplitude[i] * np.sin(2 * np.pi * freq * xtime + phase[i]) + DC[i]
    elif signal_type == "rand":
        random_points = np.random.randn(poles) * amplitude[0]  # випадкові точки
        signal = np.interp(np.linspace(0, poles, len(xtime)), np.arange(poles), random_points)  # інтерполяція для розміру xtime
    else:
        raise ValueError("Неправильний тип сигналу. Використовуйте 'sine' або 'rand'.")

    if lintrend:
        top_margin = np.max(signal)
        bot_margin = np.min(signal)
        ampl = max(2, abs(int(top_margin - bot_margin)))

        trend_line = np.random.randint(ampl - 1) + ampl
        signal += np.linspace(-trend_line, trend_line, len(xtime))

    noise = noise_amplitude * np.random.randn(len(xtime))

    signal_with_noise = signal + noise

    return xtime, signal_with_noise, signal, noise

srate = 1000
sec = 2.0
frequency = [5, 10]
amplitude = [1, 0.5]
phase = [0, np.pi/4]
DC = [0, 0.1]
lintrend = True
noise_amplitude = 0.2

xtime, signal_with_noise, original_signal, noise_template = create_signal_with_noise(
    srate, sec, "rand", noise_amplitude, frequency=frequency
)

noise_matrix = noise_template.reshape(-1, 1)
signal_with_noise_matrix = signal_with_noise.reshape(-1, 1)

weights = pinv(noise_matrix) @ signal_with_noise_matrix

estimated_noise_from_ls = noise_matrix @ weights

signal_cleaned_ls = signal_with_noise - estimated_noise_from_ls.flatten()

signal_cleaned_subtraction = signal_with_noise - noise_template

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(xtime, signal_with_noise, label="Сигнал з шумом", color='r')
plt.plot(xtime, noise_template, label="Шум", linestyle="--")
plt.title("Сигнал з шумом")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(xtime, signal_cleaned_ls, label="Сигнал після очищення (Найменші квадрати)", color='g')
plt.title("Сигнал після очищення (Найменші квадрати)")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(xtime, signal_cleaned_subtraction, label="Сигнал після очищення (Віднімання шаблону)", color='b')
plt.title("Сигнал після очищення (Віднімання шаблону)")
plt.grid(True)

plt.tight_layout()
plt.show()
