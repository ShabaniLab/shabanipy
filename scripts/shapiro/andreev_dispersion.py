import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def andreev(transmission, phase):
    return np.sqrt(1 - transmission*np.sin(phase*np.pi/2)**2)

phase = np.linspace(0, 4, 1001)

for i, t in enumerate([1, 0.995, 0.98]):
    plt.plot(phase, andreev(t, phase), color='C%d' % i, label='τ=%f' % t)
    plt.plot(phase, -andreev(t, phase), color='C%d' % i)

plt.xlabel('Phase')
plt.ylabel('Energy')
plt.legend(loc='upper right')
ax = plt.gca()
# ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%s π'))
ax.set_xticks([0, 1, 2, 3, 4])
ax.set_xticklabels(['0', 'π', '2π', '3π', '4π'])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticklabels(['-Δ', '-Δ/2', '0', 'Δ/2', 'Δ'])
plt.axhline(0.1, color='k')
plt.axhline(-0.1, color='k')
plt.annotate('', xy=(0.5, 0.1), xytext=(0.5, -0.1),
             arrowprops=dict(arrowstyle='<->'))
plt.annotate('~10 GHz', xy=(-0.15, -0.02))
plt.annotate('Δ ~ 200 µeV', xy=(0.6, 0.8))
plt.show()
