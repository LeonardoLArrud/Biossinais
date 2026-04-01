import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import neurokit2 as nk

# Exemplo de registro do PTB-XL
record_name = './data/00001_hr' 

record = wfdb.rdrecord(record_name)
fs = record.fs  # Frequência de amostragem (esperado 500 Hz)

# Extraindo a Derivação II
# O PTB-XL está organizado em (I, II, III, AVR, AVL, AVF, V1-V6)
lead_names = record.sig_name
lead_ii_idx = lead_names.index('II')
ecg_raw = record.p_signal[:, lead_ii_idx]

# Vetor de tempo em segundos
tempo = np.arange(len(ecg_raw)) / fs

def aplicar_filtros(sinal, fs):
    # Filtro Passa-Banda Butterworth - 0.5 Hz a 40 Hz
    # Remove baixas frequências (< 0.5 Hz) e ruídos de alta frequência (> 40 Hz)
    nyquist = 0.5 * fs
    low = 0.5 / nyquist
    high = 40.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    sinal_bandpass = signal.filtfilt(b, a, sinal)
    
    # Filtro Rejeita-Faixa para Ruído de Linha
    f0 = 50.0  # Frequência a ser removida
    Q = 30.0   # Fator de qualidade
    b_notch, a_notch = signal.iirnotch(f0, Q, fs)
    sinal_filtrado = signal.filtfilt(b_notch, a_notch, sinal_bandpass)
    
    return sinal_filtrado

ecg_filtrado = aplicar_filtros(ecg_raw, fs)

# Salvando num CSV para que a próxima etapa
df_saida = pd.DataFrame({'Tempo_s': tempo, 'ECG_Bruto': ecg_raw, 'ECG_Filtrado': ecg_filtrado})
df_saida.to_csv('sinal_filtrado.csv', index=False)
print("Arquivo sinal_filtrado.csv salvo com sucesso para o Passo 2!")

# 5. Gerando os Gráficos
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(tempo, ecg_raw, color='lightgray', label='Sinal Bruto')
plt.title(f'Sinal de ECG Bruto - Derivação II ({fs} Hz)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(tempo, ecg_filtrado, color='red', label='Sinal Filtrado')
plt.title('Sinal Após Filtragem')
plt.xlabel('Tempo (segundos)')
plt.ylabel('Amplitude (mV)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()