import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.0000.wav"




#waveform

# 22050 number of smaples per second taken from an analog signal to make it digital
signal, sr = librosa.load(file, sr = 22050) #Load an audio file as a floating point time series at rate sr

#signal array will have values = sr*T = 22050 * 30 (amplitude of waveform)

# librosa.display.waveplot(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()



# fft -> spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft)

frequency = np.linspace(0, sr, len(magnitude)) #linspace() -> number of evenly spaced number in an interval
left_frequency = frequency[:int(len(frequency)/2)] #frequency repeats itself after the half part therefore only half is used
left_magnitude = magnitude[:int(len(frequency)/2)]

# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()


#stft -> spectrogram
n_fft = 2048 #no. of samples per fft
hop_length = 512 #shifting the samples (column)

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram) #changing to decibal

# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()



#MFCCs
# MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
# librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()