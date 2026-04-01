import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import neurokit2 as nk

# Exemplo de registro do PTB-XL
record_name = './data500/00005_hr' 

record = wfdb.rdrecord(record_name)
fs = record.fs  # Frequência de amostragem (esperado 500 Hz)

# Extraindo a Derivação II
# O PTB-XL está organizado em (I, II, III, AVR, AVL, AVF, V1-V6)
lead_names = record.sig_name
lead_ii_idx = lead_names.index('II')
ecg_raw = record.p_signal[:, lead_ii_idx]

# Vetor de tempo em segundos
tempo = np.arange(len(ecg_raw)) / fs

sinal_filtrado = nk.ecg_clean(ecg_raw, sampling_rate=500, method="neurokit")

plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(tempo, ecg_raw, color='lightgray', label='Sinal Bruto (Com Ruído/Drift)')
plt.title(f'Sinal de ECG Bruto - Derivação II ({fs} Hz)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(tempo, sinal_filtrado, color='red', label='Sinal Filtrado (0.5-40Hz + Notch 50Hz)')
plt.title('Sinal Após Filtragem')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

qualidade = nk.ecg_quality(sinal_filtrado, sampling_rate=500, method="zhao2018")

print(qualidade)