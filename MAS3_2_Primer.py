"""
Skripta je napravljena za potrebe predmeta "Metode analize elektrofizioloskih
signala" (http://automatika.etf.bg.ac.rs/sr/13e054mas) na Elektrotehnickom 
fakultetu u Beogradu. Ideja je da se studenti upoznaju sa analizom biomedicinskih 
podataka pomoc�u programskog jezika Python.

Predstavljena je biblioteka pyphysio (https://github.com/MPBA/pyphysio).
Dokumentacija i source code su dostupni na github linku.

Biblioteku je moguc�e instalirati pomoc�u komande: pip install pyphysio.

@author: Nebojsa Jovanovic (nebojsa.php@gmail.com)
"""

import matplotlib.pyplot as plt

plt.close('all')

# 1. Generate signal
## Biblioteka pruža mogućnost korišćenja test  EKG, BVP, EDA i respiratornog
## signala. U ovom primeru će biti izgenerisan EKG signal i na njemu ce biti
## vršena analiza

from pyphysio.tests import TestData

fs = 2048 # Autori navode da su svi signali snimljeni sa fs = 2048 Hz
ecg_data = TestData.ecg()

# Osnovne klase ove biblioteke jesu EvenlySignal (za const fs) i 
# UnevenlySignal (kada fs nije const). Sve signale koji su dati kao numpy
# niz je potrebno pretvoriti u objekat jedne od ovih klasa za dalju obradu
# Za više informacija pogledati source code

from pyphysio import EvenlySignal
ecg = EvenlySignal(values = ecg_data, 
                   sampling_freq = fs,
                   signal_type = 'ECG')

# Frekvencija odabiranja za EKG signal je prevelika, zato ce u prvom koraku
# signal biti downsamplovan na 128 Hz
ecg_128 = ecg.resample(128)

# Prikaz citavog EKG signala
plt.figure()
ecg_128.plot()
plt.xlabel('Vreme [s]')

# Segmentacija EKG signala do 10. sekunde i prikaz tog dela

ecg_segmented = ecg_128.segment_time(0,10)
plt.figure()
ecg_segmented.plot()
plt.xlabel('Vreme [s]')


# Prikaz spektralne gustine EKG signala
# Kada se primenjuje bilo koji alat za analizu prvo je potrebno napraviti
# objekat klase i zatim primeniti analizu kao objekat(signal). Moguce je i 
# direktno po konstrukciji objekta to izvrsiti da se ubrza proces kodiranja.
# Bice pokazana oba nacina 

from pyphysio.tools.Tools import PSD

# 1. nacin
# Kreira se objekat klase PSD

psd_spect = PSD(method = 'fft', nfft = 2**14)
f_osa, PSD = psd_spect(ecg_128)

plt.figure()
plt.plot(f_osa,PSD)
plt.xlabel('Frekvencija [Hz]')
plt.ylabel('Magnituda [mV^2/Hz]')


# 2. Filtriranje EKG signala, filtrom propusnikom visokih ucestanosti

from pyphysio.filters.Filters import IIRFilter

# 2.nacin 
# Direktno se poziva
ecg_filt = IIRFilter(fp = 4, fs = 2, att = 20)(ecg_128)

f_osa, PSD2 = psd_spect(ecg_filt)

plt.figure()
plt.subplot(2,1,1)
ecg_filt.plot()
plt.xlabel('Vreme [s]')
plt.ylabel('Amplituda [mV]')
plt.subplot(2,1,2)
plt.plot(f_osa,PSD2)
plt.xlabel('Frekvencija [Hz]')
plt.ylabel('Magnituda [mV^2/Hz]')


# 3. Nalazenje R pikova signala

from pyphysio.tools.Tools import PeakDetection

indexes,_,values,_ = PeakDetection(delta = 0.8, refractory = 0.25)(ecg_segmented)

plt.figure()
ecg_segmented.plot()
plt.plot(indexes/128,values,'rx',label = 'Pikovi')
plt.xlabel('Vreme [s]')
plt.ylabel('Amplituda [mV]')
plt.legend(loc = 'upper right')

# 4. Racunanje IBI (Interbeat intervala)

from pyphysio.estimators.Estimators import BeatFromECG

ibi = BeatFromECG(bpm_max = 100)(ecg_128)

plt.figure()
ibi.plot()
plt.xlabel('Vreme [s]')
plt.ylabel('IBI [s]')


