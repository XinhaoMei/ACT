#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import torch
import librosa
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tools.file_io import load_pickle_file


class AudioCapsDataset(Dataset):

    def __init__(self, config):
        super(AudioCapsDataset, self).__init__()

        self.h5_path = 'data/hdf5s/train/train.h5'
        vocabulary_path = 'data/pickles/words_list.p'
        with h5py.File(self.h5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.captions = [caption.decode() for caption in hf['caption'][:]]

        self.vocabulary = load_pickle_file(vocabulary_path)

        self.sr = config.wav.sr
        self.window_length = config.wav.window_length
        self.hop_length = config.wav.hop_length
        self.n_mels = config.wav.n_mels

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, index):

        with h5py.File(self.h5_path, 'r') as hf:
            waveform = self.resample(hf['waveform'][index])
        audio_name = self.audio_names[index]
        caption = self.captions[index]

        feature = librosa.feature.melspectrogram(waveform, sr=self.sr, n_fft=self.window_length,
                                                 hop_length=self.hop_length, n_mels=self.n_mels)
        feature = librosa.power_to_db(feature).T
        feature = feature[:-1, :]
        words = caption.strip().split()
        target = np.array([self.vocabulary.index(word) for word in words])
        target_len = len(target)

        return feature, target, target_len, audio_name, caption

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sr == 32000:
            return waveform
        elif self.sr == 16000:
            return waveform[0:: 2]
        elif self.sr == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')


class AudioCapsEvalDataset(Dataset):

    def __init__(self, split, config):

        if split == 'val':
            self.h5_path = 'data/hdf5s/val/val.h5'
        elif split == 'test':
            self.h5_path = 'data/hdf5s/test/test.h5'
        with h5py.File(self.h5_path, 'r') as hf:
            self.audio_names = [audio_name.decode() for audio_name in hf['audio_name'][:]]
            self.captions = [caption for caption in hf['caption'][:]]

        self.sr = config.wav.sr
        self.window_length = config.wav.window_length
        self.hop_length = config.wav.hop_length
        self.n_mels = config.wav.n_mels

        self.caption_field = ['caption_{}'.format(i) for i in range(1, 6)]

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, index):
        with h5py.File(self.h5_path, 'r') as hf:
            waveform = self.resample(hf['waveform'][index])
        audio_name = self.audio_names[index]
        captions = self.captions[index]

        target_dict = {}
        for i, cap_ind in enumerate(self.caption_field):
            target_dict[cap_ind] = captions[i].decode()

        feature = librosa.feature.melspectrogram(waveform, sr=self.sr, n_fft=self.window_length,
                                                 hop_length=self.hop_length, n_mels=self.n_mels)
        feature = librosa.power_to_db(feature).T
        feature = feature[:-1, :]

        return feature, target_dict, audio_name

    def resample(self, waveform):
        """Resample.
        Args:
          waveform: (clip_samples,)
        Returns:
          (resampled_clip_samples,)
        """
        if self.sr == 32000:
            return waveform
        elif self.sr == 16000:
            return waveform[0:: 2]
        elif self.sr == 8000:
            return waveform[0:: 4]
        else:
            raise Exception('Incorrect sample rate!')


def get_audiocaps_loader(split,
                         config):
    if split == 'train':
        dataset = AudioCapsDataset(config)
        return DataLoader(dataset=dataset, batch_size=config.data.batch_size,
                          shuffle=True, drop_last=True,
                          num_workers=config.data.num_workers, collate_fn=collate_fn)
    elif split in ['val', 'test']:
        dataset = AudioCapsEvalDataset(split, config)
        return DataLoader(dataset=dataset, batch_size=config.data.batch_size,
                          shuffle=False, drop_last=False,
                          num_workers=config.data.num_workers, collate_fn=collate_fn_eval)


def collate_fn(batch):

    max_caption_length = max(i[1].shape[0] for i in batch)

    eos_token = batch[0][1][-1]

    words_tensor = []

    for _, words_indexs, _, _, _ in batch:
        if max_caption_length >= words_indexs.shape[0]:
            padding = torch.ones(max_caption_length - len(words_indexs)).mul(eos_token).long()
            data = [torch.from_numpy(words_indexs).long(), padding]
            tmp_words_indexs = torch.cat(data)
        else:
            tmp_words_indexs = torch.from_numpy(words_indexs[:max_caption_length]).long()
        words_tensor.append(tmp_words_indexs.unsqueeze_(0))

    feature = [i[0] for i in batch]
    feature_tensor = torch.tensor(feature)
    target_tensor = torch.cat(words_tensor)

    target_lens = [i[2] for i in batch]
    file_names = [i[3] for i in batch]
    captions = [i[4] for i in batch]

    return feature_tensor, target_tensor, target_lens, file_names, captions


def collate_fn_eval(batch):

    feature = [i[0] for i in batch]
    feature_tensor = torch.tensor(feature)

    file_names = [i[2] for i in batch]
    target_dicts = [i[1] for i in batch]

    return feature_tensor, target_dicts, file_names
