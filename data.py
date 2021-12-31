import os

import librosa

PATH = os.path.dirname(os.path.realpath(__file__))
LIBRISPEECH_SAMPLING_RATE = 16000
HOP_LENGTH = 512 #the default spacing between frames
N_FFT = 255 #number of samples


from tqdm import tqdm
import torch.utils.data
import soundfile as sf
import pandas as pd
import numpy as np
import json
import os


sex_to_label = {'M': 0, 'F': 1}
label_to_sex = {0: 'M', 1: 'F'}


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, subsets, length, stochastic=True, cache=True):
        """
        This class subclasses the torch Dataset object. The __getitem__ function will return a raw audio sample and it's
        label.
        :param subsets: What LibriSpeech datasets to use
        :param length: Number of audio samples to take from each file. Any files shorter than this will be ignored.
        :param stochastic: If True then we will take a random fragment from each file of sufficient length. If False we
        wil always take a fragment starting at the beginning of a file.
        """
        assert isinstance(length, (int)), 'Length is not an integer!'
        self.subset = subsets
        self.fragment_length = length
        self.stochastic = stochastic

        print('Initialising LibriSpeechDataset with length = {} and subsets = {}'.format(length, subsets))

        # Convert subset to list if it is a string
        # This allows to handle list of multiple subsets the same a single subset
        if isinstance(subsets, str):
            subsets = [subsets]

        # Check if we have already indexed the files
        cached_id_to_filepath_location = '/data/LibriSpeech__datasetid_to_filepath__subsets={}__length={}.json'.format(
            subsets, length)
        cached_id_to_filepath_location = PATH + cached_id_to_filepath_location

        cached_id_to_sex_location = '/data/LibriSpeech__datasetid_to_sex__subsets={}__length={}.json'.format(
            subsets, length)
        cached_id_to_sex_location = PATH + cached_id_to_sex_location

        cached_dictionaries_exist = os.path.exists(cached_id_to_filepath_location) \
            and os.path.exists(cached_id_to_sex_location)
        if cache and cached_dictionaries_exist:
            print('Cached indexes found.')
            with open(cached_id_to_filepath_location) as f:
                self.datasetid_to_filepath = json.load(f)

            with open(cached_id_to_sex_location) as f:
                self.datasetid_to_sex = json.load(f)

            # The dictionaries loaded from json have string type keys
            # Convert them back to integers
            self.datasetid_to_filepath = {int(k): v for k, v in self.datasetid_to_filepath.items()}
            self.datasetid_to_sex = {int(k): v for k, v in self.datasetid_to_sex.items()}

            assert len(self.datasetid_to_filepath) == len(self.datasetid_to_sex), 'Cached indexes are different lengths!'

            self.n_files = len(self.datasetid_to_filepath)
            print('{} usable files found.'.format(self.n_files))

            return

        df = pd.read_csv(PATH+'/data/LibriSpeech/SPEAKERS.TXT', skiprows=11, delimiter='|', error_bad_lines=False)
        df.columns = [col.strip().replace(';', '').lower() for col in df.columns]
        df = df.assign(
            sex=df['sex'].apply(lambda x: x.strip()),
            subset=df['subset'].apply(lambda x: x.strip()),
            name=df['name'].apply(lambda x: x.strip()),
        )

        # Get id -> sex mapping
        librispeech_id_to_sex = df[df['subset'].isin(subsets)][['id', 'sex']].to_dict()
        self.librispeech_id_to_sex = {
            k: v for k, v in zip(librispeech_id_to_sex['id'].values(), librispeech_id_to_sex['sex'].values())}
        librispeech_id_to_name = df[df['subset'].isin(subsets)][['id', 'name']].to_dict()
        self.librispeech_id_to_name = {
            k: v for k, v in zip(librispeech_id_to_name['id'].values(), librispeech_id_to_name['name'].values())}

        datasetid = 0
        self.n_files = 0
        self.datasetid_to_filepath = {}
        self.datasetid_to_sex = {}
        self.datasetid_to_name = {}

        for s in subsets:
            print('Indexing {}...'.format(s))
            # Quick first pass to find total for tqdm bar
            subset_len = 0
            for root, folders, files in os.walk(PATH+'/data/LibriSpeech/{}/'.format(s)):
                subset_len += len([f for f in files if f.endswith('.flac')])

            progress_bar = tqdm(total=subset_len)
            for root, folders, files in os.walk(os.path.join(PATH,"data","LibriSpeech",s)):
                if len(files) == 0:
                    continue
                print(root)
                librispeech_id = int(root.split('\\')[-2])

                for f in files:
                    # Skip non-sound files
                    if not f.endswith('.flac'):
                        continue

                    progress_bar.update(1)

                    # Skip short files
                    instance, samplerate = sf.read(os.path.join(root, f))
                    if len(instance) <= self.fragment_length:
                        continue

                    self.datasetid_to_filepath[datasetid] = os.path.abspath(os.path.join(root, f))
                    self.datasetid_to_sex[datasetid] = self.librispeech_id_to_sex[librispeech_id]
                    self.datasetid_to_name[datasetid] = self.librispeech_id_to_name[librispeech_id]
                    datasetid += 1
                    self.n_files += 1

            progress_bar.close()
        print('Finished indexing data. {} usable files found.'.format(self.n_files))

        # Save relevant dictionaries to json in order to re-use them layer
        # The indexing takes a few minutes each time and would be nice to just perform this calculation once
        with open(cached_id_to_filepath_location, 'w') as f:
            json.dump(self.datasetid_to_filepath, f)

        with open(cached_id_to_sex_location, 'w') as f:
            json.dump(self.datasetid_to_sex, f)

    def __getitem__(self, index):
        features = []  # list to save features
        labels = []  # list to save labels

        file_path = self.datasetid_to_filepath[index]
        librosa_audio_data, sample_rate= librosa.load(file_path, sr=LIBRISPEECH_SAMPLING_RATE)

         # = sf.read(file_path)
        # Choose a random sample of the file
        if self.stochastic:
            fragment_start_index = np.random.randint(0, len(librosa_audio_data)-self.fragment_length)
        else:
            fragment_start_index = 0

        # cut the file to wanted length
        # y_cut = y[round(tstart * sr, ndigits=None) //TODO
        #           :round(tend * sr, ndigits=None)]
        librosa_audio_data = librosa_audio_data[fragment_start_index:fragment_start_index+self.fragment_length]
        # data = librosa.feature.mfcc(instance,n_fft=N_FFT , hop_length=HOP_LENGTH, n_mfcc=128)
        mfccs_features=np.array(librosa.feature.mfcc(librosa_audio_data, sr=sample_rate, n_mfcc=128))
        # mfccs_scaled_features=np.mean(mfccs_features.T,axis=0)
        sex = self.datasetid_to_sex[index]
        return mfccs_features, sex_to_label[sex]

    def __len__(self):
        return self.n_files

# def get_features(df_in):
#     features=[] #list to save features
#     labels=[] #list to save labels
#     for index in range(0,len(df_in)):
#       #get the filename
#       filename = df_in.iloc[index]['recording_id']+str('.flac')
#       #cut to start of signal
#       tstart = df_in.iloc[index]['t_min']
#       #cut to end of signal
#       tend = df_in.iloc[index]['t_max']
#       #save labels
#       species_id = df_in.iloc[index]['species_id']
#       #load the file
#       y, sr = librosa.load('train/'+filename,sr=28000)
#       #cut the file from tstart to tend
#       y_cut = y[round(tstart*sr,ndigits=None)
#          :round(tend*sr, ndigits= None)]
#       data = np.array([padding(librosa.feature.mfcc(y_cut,
#          n_fft=n_fft,hop_length=hop_length,n_mfcc=128),1,400)])
#       features.append(data)
#       labels.append(species_id)
#     output=np.concatenate(features,axis=0)
#     return(np.array(output), labels)
