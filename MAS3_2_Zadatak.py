"""
Skripta je napravljena za potrebe predmeta "Metode analize elektrofizioloskih
signala" (http://automatika.etf.bg.ac.rs/sr/13e054mas) na Elektrotehnickom 
fakultetu u Beogradu. Ideja je da se studenti upoznaju sa analizom biomedicinskih 
podataka pomocu programskog jezika Python.

Predstavljena je biblioteka pyphysio (https://github.com/MPBA/pyphysio).
Dokumentacija i source code su dostupni na github linku.

Biblioteku je moguce instalirati pomocu komande: pip install pyphysio.

@author: Nebojsa Jovanovic (nebojsa.php@gmail.com)
"""


# 1. Ucitavanje EKG signala i njegovo pretprocesiranje

import matplotlib.pyplot as plt
from pyphysio.tests import TestData
from pyphysio import EvenlySignal

plt.close('all')

fs = 2048 
ecg_data = TestData.ecg()
ecg = EvenlySignal(values = ecg_data, 
                   sampling_freq = fs,
                   signal_type = 'ECG')

ecg_128 = ecg.resample(128)

from pyphysio.filters.Filters import IIRFilter

ecg_filt = IIRFilter(fp = [4,45], fs = [2,48], loss = .5, att = 20)(ecg_128)

# Racunanje FFT-a

import scipy.fftpack as fft
import numpy as np

fftsig = np.abs(fft.fft(ecg_filt,2**14))
fftsig = fftsig[range(2**14//2)]
f_osa = np.linspace(0,128/2,len(fftsig))

plt.figure()
plt.plot(f_osa,fftsig)
plt.xlabel('Frekvencija [Hz]')
plt.ylabel('Magnituda [a.u.]')

# 2. Racunanje HRV parametra


