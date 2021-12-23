import pandas as pd
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.layers import LSTM, Dense

# sample_num=3 #pick a file to display
filename= "6467-94831-0000.flac"
#define the beginning time of the signal
tstart = 0.1 
tend =0.6 #define the end time of the signal
y,sr=librosa.load(str(filename))
# print("y: "+y)
librosa.display.waveplot(y,sr=sr, x_axis='time', color='purple',offset=0.0)
hop_length = 512 #the default spacing between frames
n_fft = 255 #number of samples 
#cut the sample to the relevant times
y_cut=y[int(round(tstart*sr)):int(round(tend*sr))]
MFCCs = librosa.feature.mfcc(y_cut, n_fft=n_fft,n_mfcc=13)
print(MFCCs.shape)
fig, ax = plt.subplots(figsize=(20,7))
# librosa.display.specshow(MFCCs,sr=sr, cmap='cool',hop_length=hop_length)
# ax.set_xlabel('Time', fontsize=15)
# ax.set_title('MFCC', size=20)
# plt.colorbar()
# plt.show()

# sample_num=3 #pick a file to display
filename= "6467-94831-0012.flac"
#define the beginning time of the signal
tstart = 0.1 
tend =0.6 #define the end time of the signal
y,sr=librosa.load(str(filename))
# print("y: "+y)
librosa.display.waveplot(y,sr=sr, x_axis='time', color='purple',offset=0.0)
hop_length = 512 #the default spacing between frames
n_fft = 255 #number of samples 
#cut the sample to the relevant times
y_cut=y[int(round(tstart*sr)):int(round(tend*sr))]
MFCCs = librosa.feature.mfcc(y_cut, n_fft=n_fft,n_mfcc=13)
print(MFCCs.shape)
fig, ax = plt.subplots(figsize=(20,7))
# librosa.display.specshow(MFCCs,sr=sr, cmap='cool',hop_length=hop_length)
# ax.set_xlabel('Time', fontsize=15)
# ax.set_title('MFCC', size=20)
# plt.colorbar()
# plt.show()