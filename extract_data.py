import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
audio_file= "6467-94831-0012.flac"
# ipd.Audio(audio_file)
# load audio files with librosa
signal, sr = librosa.load(audio_file)
#Extracting MFCCs
mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
mfccs.shape
print(mfccs.shape)
#Visualising MFCCs
plt.figure(figsize=(25, 10))
librosa.display.specshow(mfccs, 
                         x_axis="time", 
                         sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
