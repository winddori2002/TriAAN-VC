# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Utility functions."""

import fnmatch
import logging
import os
import sys
import tarfile

from distutils.version import LooseVersion
from filelock import FileLock

import h5py
import numpy as np
import torch
import yaml

PRETRAINED_MODEL_LIST = {
    "ljspeech_parallel_wavegan.v1": "1PdZv37JhAQH6AwNh31QlqruqrvjTBq7U",
    "ljspeech_parallel_wavegan.v1.long": "1A9TsrD9fHxFviJVFjCk5W6lkzWXwhftv",
    "ljspeech_parallel_wavegan.v1.no_limit": "1CdWKSiKoFNPZyF1lo7Dsj6cPKmfLJe72",
    "ljspeech_parallel_wavegan.v3": "1-oZpwpWZMMolDYsCqeL12dFkXSBD9VBq",
    "ljspeech_melgan.v1": "1i7-FPf9LPsYLHM6yNPoJdw5Q9d28C-ip",
    "ljspeech_melgan.v1.long": "1x1b_R7d2561nqweK3FPb2muTdcFIYTu6",
    "ljspeech_melgan.v3": "1J5gJ_FUZhOAKiRFWiAK6FcO5Z6oYJbmQ",
    "ljspeech_melgan.v3.long": "124JnaLcRe7TsuAGh3XIClS3C7Wom9AU2",
    "ljspeech_full_band_melgan.v2": "1Kb7q5zBeQ30Wsnma0X23G08zvgDG5oen",
    "ljspeech_multi_band_melgan.v2": "1b70pJefKI8DhGYz4SxbEHpxm92tj1_qC",
    "ljspeech_hifigan.v1": "1i6-hR_ksEssCYNlNII86v3AoeA1JcuWD",
    "ljspeech_style_melgan.v1": "10aJSZfmCAobQJgRGio6cNyw6Xlgmme9-",
    "jsut_parallel_wavegan.v1": "1qok91A6wuubuz4be-P9R2zKhNmQXG0VQ",
    "jsut_multi_band_melgan.v2": "1chTt-76q2p69WPpZ1t1tt8szcM96IKad",
    "jsut_hifigan.v1": "1vdgqTu9YKyGMCn-G7H2fI6UBC_4_55XB",
    "jsut_style_melgan.v1": "1VIkjSxYxAGUVEvJxNLaOaJ7Twe48SH-s",
    "csmsc_parallel_wavegan.v1": "1QTOAokhD5dtRnqlMPTXTW91-CG7jf74e",
    "csmsc_multi_band_melgan.v2": "1G6trTmt0Szq-jWv2QDhqglMdWqQxiXQT",
    "csmsc_hifigan.v1": "1fVKGEUrdhGjIilc21Sf0jODulAq6D1qY",
    "csmsc_style_melgan.v1": "1kGUC_b9oVSv24vZRi66AAbSNUKJmbSCX",
    "arctic_slt_parallel_wavegan.v1": "1_MXePg40-7DTjD0CDVzyduwQuW_O9aA1",
    "jnas_parallel_wavegan.v1": "1D2TgvO206ixdLI90IqG787V6ySoXLsV_",
    "vctk_parallel_wavegan.v1": "1bqEFLgAroDcgUy5ZFP4g2O2MwcwWLEca",
    "vctk_parallel_wavegan.v1.long": "1tO4-mFrZ3aVYotgg7M519oobYkD4O_0-",
    "vctk_multi_band_melgan.v2": "10PRQpHMFPE7RjF-MHYqvupK9S0xwBlJ_",
    "vctk_hifigan.v1": "1oVOC4Vf0DYLdDp4r7GChfgj7Xh5xd0ex",
    "vctk_style_melgan.v1": "14ThSEgjvl_iuFMdEGuNp7d3DulJHS9Mk",
    "libritts_parallel_wavegan.v1": "1zHQl8kUYEuZ_i1qEFU6g2MEu99k3sHmR",
    "libritts_parallel_wavegan.v1.long": "1b9zyBYGCCaJu0TIus5GXoMF8M3YEbqOw",
    "libritts_multi_band_melgan.v2": "1kIDSBjrQvAsRewHPiFwBZ3FDelTWMp64",
    "libritts_hifigan.v1": "1_TVFIvVtMn-Z4NiQrtrS20uSJOvBsnu1",
    "libritts_style_melgan.v1": "1yuQakiMP0ECdB55IoxEGCbXDnNkWCoBg",
    "kss_parallel_wavegan.v1": "1mLtQAzZHLiGSWguKCGG0EZa4C_xUO5gX",
    "hui_acg_hokuspokus_parallel_wavegan.v1": "1irKf3okMLau56WNeOnhr2ZfSVESyQCGS",
    "ruslan_parallel_wavegan.v1": "1M3UM6HN6wrfSe5jdgXwBnAIl_lJzLzuI",
}


def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files


def read_hdf5(hdf5_name, hdf5_path):
    """Read hdf5 dataset.

    Args:
        hdf5_name (str): Filename of hdf5 file.
        hdf5_path (str): Dataset name in hdf5 file.

    Return:
        any: Dataset values.

    """
    if not os.path.exists(hdf5_name):
        logging.error(f"There is no such a hdf5 file ({hdf5_name}).")
        sys.exit(1)

    hdf5_file = h5py.File(hdf5_name, "r")

    if hdf5_path not in hdf5_file:
        logging.error(f"There is no such a data in hdf5 file. ({hdf5_path})")
        sys.exit(1)

    hdf5_data = hdf5_file[hdf5_path][()]
    hdf5_file.close()

    return hdf5_data


def write_hdf5(hdf5_name, hdf5_path, write_data, is_overwrite=True):
    """Write dataset to hdf5.

    Args:
        hdf5_name (str): Hdf5 dataset filename.
        hdf5_path (str): Dataset path in hdf5.
        write_data (ndarray): Data to write.
        is_overwrite (bool): Whether to overwrite dataset.

    """
    # convert to numpy array
    write_data = np.array(write_data)

    # check folder existence
    folder_name, _ = os.path.split(hdf5_name)
    if not os.path.exists(folder_name) and len(folder_name) != 0:
        os.makedirs(folder_name)

    # check hdf5 existence
    if os.path.exists(hdf5_name):
        # if already exists, open with r+ mode
        hdf5_file = h5py.File(hdf5_name, "r+")
        # check dataset existence
        if hdf5_path in hdf5_file:
            if is_overwrite:
                logging.warning(
                    "Dataset in hdf5 file already exists. " "recreate dataset in hdf5."
                )
                hdf5_file.__delitem__(hdf5_path)
            else:
                logging.error(
                    "Dataset in hdf5 file already exists. "
                    "if you want to overwrite, please set is_overwrite = True."
                )
                hdf5_file.close()
                sys.exit(1)
    else:
        # if not exists, open with w mode
        hdf5_file = h5py.File(hdf5_name, "w")

    # write data to hdf5
    hdf5_file.create_dataset(hdf5_path, data=write_data)
    hdf5_file.flush()
    hdf5_file.close()


class HDF5ScpLoader(object):
    """Loader class for a fests.scp file of hdf5 file.

    Examples:
        key1 /some/path/a.h5:feats
        key2 /some/path/b.h5:feats
        key3 /some/path/c.h5:feats
        key4 /some/path/d.h5:feats
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp")
        >>> array = loader["key1"]

        key1 /some/path/a.h5
        key2 /some/path/b.h5
        key3 /some/path/c.h5
        key4 /some/path/d.h5
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp", "feats")
        >>> array = loader["key1"]

        key1 /some/path/a.h5:feats_1,feats_2
        key2 /some/path/b.h5:feats_1,feats_2
        key3 /some/path/c.h5:feats_1,feats_2
        key4 /some/path/d.h5:feats_1,feats_2
        ...
        >>> loader = HDF5ScpLoader("hdf5.scp")
        # feats_1 and feats_2 will be concatenated
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp, default_hdf5_path="feats"):
        """Initialize HDF5 scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with hdf5 format.
            default_hdf5_path (str): Path in hdf5 file. If the scp contain the info, not used.

        """
        self.default_hdf5_path = default_hdf5_path
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get hdf5 file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        p = self.data[key]
        if ":" in p:
            if len(p.split(",")) == 1:
                return read_hdf5(*p.split(":"))
            else:
                p1, p2 = p.split(":")
                feats = [read_hdf5(p1, p) for p in p2.split(",")]
                return np.concatenate(
                    [f if len(f.shape) != 1 else f.reshape(-1, 1) for f in feats], 1
                )
        else:
            return read_hdf5(p, self.default_hdf5_path)

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]


class NpyScpLoader(object):
    """Loader class for a fests.scp file of npy file.

    Examples:
        key1 /some/path/a.npy
        key2 /some/path/b.npy
        key3 /some/path/c.npy
        key4 /some/path/d.npy
        ...
        >>> loader = NpyScpLoader("feats.scp")
        >>> array = loader["key1"]

    """

    def __init__(self, feats_scp):
        """Initialize npy scp loader.

        Args:
            feats_scp (str): Kaldi-style feats.scp file with npy format.

        """
        with open(feats_scp) as f:
            lines = [line.replace("\n", "") for line in f.readlines()]
        self.data = {}
        for line in lines:
            key, value = line.split()
            self.data[key] = value

    def get_path(self, key):
        """Get npy file path for a given key."""
        return self.data[key]

    def __getitem__(self, key):
        """Get ndarray for a given key."""
        return np.load(self.data[key])

    def __len__(self):
        """Return the length of the scp file."""
        return len(self.data)

    def __iter__(self):
        """Return the iterator of the scp file."""
        return iter(self.data)

    def keys(self):
        """Return the keys of the scp file."""
        return self.data.keys()

    def values(self):
        """Return the values of the scp file."""
        for key in self.keys():
            yield self[key]


def load_model(checkpoint, config=None, stats=None):
    """Load trained model.

    Args:
        checkpoint (str): Checkpoint path.
        config (dict): Configuration dict.
        stats (str): Statistics file path.

    Return:
        torch.nn.Module: Model instance.

    """
    # load config if not provided
    if config is None:
        dirname = os.path.dirname(checkpoint)
        config = os.path.join(dirname, "config.yml")
        with open(config) as f:
            config = yaml.load(f, Loader=yaml.Loader)

    # lazy load for circular error
    import parallel_wavegan.models

    # get model and load parameters
    model_class = getattr(
        parallel_wavegan.models,
        config.get("generator_type", "ParallelWaveGANGenerator"),
    )
    # workaround for typo #295
    generator_params = {
        k.replace("upsample_kernal_sizes", "upsample_kernel_sizes"): v
        for k, v in config["generator_params"].items()
    }
    model = model_class(**generator_params)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu")["model"]["generator"]
    )

    # check stats existence
    if stats is None:
        dirname = os.path.dirname(checkpoint)
        if config["format"] == "hdf5":
            ext = "h5"
        else:
            ext = "npy"
        if os.path.exists(os.path.join(dirname, f"stats.{ext}")):
            stats = os.path.join(dirname, f"stats.{ext}")

    # load stats
    if stats is not None:
        model.register_stats(stats)

    # add pqmf if needed
    if config["generator_params"]["out_channels"] > 1:
        # lazy load for circular error
        from parallel_wavegan.layers import PQMF

        pqmf_params = {}
        if LooseVersion(config.get("version", "0.1.0")) <= LooseVersion("0.4.2"):
            # For compatibility, here we set default values in version <= 0.4.2
            pqmf_params.update(taps=62, cutoff_ratio=0.15, beta=9.0)
        model.pqmf = PQMF(
            subbands=config["generator_params"]["out_channels"],
            **config.get("pqmf_params", pqmf_params),
        )

    return model


def download_pretrained_model(tag, download_dir=None):
    """Download pretrained model form google drive.

    Args:
        tag (str): Pretrained model tag.
        download_dir (str): Directory to save downloaded files.

    Returns:
        str: Path of downloaded model checkpoint.

    """
    assert tag in PRETRAINED_MODEL_LIST, f"{tag} does not exists."
    id_ = PRETRAINED_MODEL_LIST[tag]
    if download_dir is None:
        download_dir = os.path.expanduser("~/.cache/parallel_wavegan")
    output_path = f"{download_dir}/{tag}.tar.gz"
    os.makedirs(f"{download_dir}", exist_ok=True)
    with FileLock(output_path + ".lock"):
        if not os.path.exists(output_path):
            # lazy load for compatibility
            import gdown

            gdown.download(
                f"https://drive.google.com/uc?id={id_}", output_path, quiet=False
            )
            with tarfile.open(output_path, "r:*") as tar:
                for member in tar.getmembers():
                    if member.isreg():
                        member.name = os.path.basename(member.name)
                        tar.extract(member, f"{download_dir}/{tag}")
    checkpoint_path = find_files(f"{download_dir}/{tag}", "checkpoint*.pkl")

    return checkpoint_path[0]

# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules."""

import logging
import os

from multiprocessing import Manager

import numpy as np

from torch.utils.data import Dataset


class AudioMelDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*.h5",
        mel_query="*.h5",
        audio_load_fn=lambda x: read_hdf5(x, "wave"),
        mel_load_fn=lambda x: read_hdf5(x, "feats"),
        audio_length_threshold=None,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            mel_query (str): Query to find feature files in root_dir.
            audio_load_fn (func): Function to load audio file.
            mel_load_fn (func): Function to load feature file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."
        assert len(audio_files) == len(
            mel_files
        ), f"Number of audio and mel files are different ({len(audio_files)} vs {len(mel_files)})."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.mel_load_fn = mel_load_fn
        self.mel_files = mel_files
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio signal (T,).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])
        mel = self.mel_load_fn(self.mel_files[idx])

        if self.return_utt_id:
            items = utt_id, audio, mel
        else:
            items = audio, mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class AudioDataset(Dataset):
    """PyTorch compatible audio dataset."""

    def __init__(
        self,
        root_dir,
        audio_query="*-wave.npy",
        audio_length_threshold=None,
        audio_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            audio_query (str): Query to find audio files in root_dir.
            audio_load_fn (func): Function to load audio file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of audio and mel files
        audio_files = sorted(find_files(root_dir, audio_query))

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio_load_fn(f).shape[0] for f in audio_files]
            idxs = [
                idx
                for idx in range(len(audio_files))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_files) != len(idxs):
                logging.waning(
                    f"some files are filtered by audio length threshold "
                    f"({len(audio_files)} -> {len(idxs)})."
                )
            audio_files = [audio_files[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_files) != 0, f"Not found any audio files in ${root_dir}."

        self.audio_files = audio_files
        self.audio_load_fn = audio_load_fn
        self.return_utt_id = return_utt_id
        if ".npy" in audio_query:
            self.utt_ids = [
                os.path.basename(f).replace("-wave.npy", "") for f in audio_files
            ]
        else:
            self.utt_ids = [
                os.path.splitext(os.path.basename(f))[0] for f in audio_files
            ]
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(audio_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Audio (T,).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        audio = self.audio_load_fn(self.audio_files[idx])

        if self.return_utt_id:
            items = utt_id, audio
        else:
            items = audio

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)


class MelDataset(Dataset):
    """PyTorch compatible mel dataset."""

    def __init__(
        self,
        root_dir,
        mel_query="*-feats.npy",
        mel_length_threshold=None,
        mel_load_fn=np.load,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            mel_query (str): Query to find feature files in root_dir.
            mel_load_fn (func): Function to load feature file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return the utterance id with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # find all of the mel files
        mel_files = sorted(find_files(root_dir, mel_query))

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel_load_fn(f).shape[0] for f in mel_files]
            idxs = [
                idx
                for idx in range(len(mel_files))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_files) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_files)} -> {len(idxs)})."
                )
            mel_files = [mel_files[idx] for idx in idxs]

        # assert the number of files
        assert len(mel_files) != 0, f"Not found any mel files in ${root_dir}."

        self.mel_files = mel_files
        self.mel_load_fn = mel_load_fn
        self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        if ".npy" in mel_query:
            self.utt_ids = [
                os.path.basename(f).replace("-feats.npy", "") for f in mel_files
            ]
        else:
            self.utt_ids = [os.path.splitext(os.path.basename(f))[0] for f in mel_files]
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache
        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(mel_files))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        mel = self.mel_load_fn(self.mel_files[idx])

        if self.return_utt_id:
            items = utt_id, mel
        else:
            items = mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.mel_files)


# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules based on kaldi-style scp files."""

import logging

from multiprocessing import Manager

import kaldiio
import numpy as np

from torch.utils.data import Dataset

def _get_feats_scp_loader(feats_scp):
    # read the first line of feats.scp file
    with open(feats_scp) as f:
        key, value = f.readlines()[0].replace("\n", "").split()

    # check scp type
    if ":" in value:
        value_1, value_2 = value.split(":")
        if value_1.endswith(".ark"):
            # kaldi-ark case: utt_id_1 /path/to/utt_id_1.ark:index
            return kaldiio.load_scp(feats_scp)
        elif value_1.endswith(".h5"):
            # hdf5 case with path in hdf5: utt_id_1 /path/to/utt_id_1.h5:feats
            return HDF5ScpLoader(feats_scp)
        else:
            raise ValueError("Not supported feats.scp type.")
    else:
        if value.endswith(".h5"):
            # hdf5 case without path in hdf5: utt_id_1 /path/to/utt_id_1.h5
            return HDF5ScpLoader(feats_scp)
        elif value.endswith(".npy"):
            # npy case: utt_id_1 /path/to/utt_id_1.npy
            return NpyScpLoader(feats_scp)
        else:
            raise ValueError("Not supported feats.scp type.")


class AudioMelSCPDataset(Dataset):
    """PyTorch compatible audio and mel dataset based on kaldi-stype scp files."""

    def __init__(
        self,
        wav_scp,
        feats_scp,
        segments=None,
        audio_length_threshold=None,
        mel_length_threshold=None,
        return_utt_id=False,
        return_sampling_rate=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            wav_scp (str): Kaldi-style wav.scp file.
            feats_scp (str): Kaldi-style fests.scp file.
            segments (str): Kaldi-style segments file.
            audio_length_threshold (int): Threshold to remove short audio files.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return utterance id.
            return_sampling_rate (bool): Wheter to return sampling rate.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load scp as lazy dict
        audio_loader = kaldiio.load_scp(wav_scp, segments=segments)
        mel_loader = _get_feats_scp_loader(feats_scp)
        audio_keys = list(audio_loader.keys())
        mel_keys = list(mel_loader.keys())

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio.shape[0] for _, audio in audio_loader.values()]
            idxs = [
                idx
                for idx in range(len(audio_keys))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_keys) != len(idxs):
                logging.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_keys)} -> {len(idxs)})."
                )
            audio_keys = [audio_keys[idx] for idx in idxs]
            mel_keys = [mel_keys[idx] for idx in idxs]
        if mel_length_threshold is not None:
            mel_lengths = [mel.shape[0] for mel in mel_loader.values()]
            idxs = [
                idx
                for idx in range(len(mel_keys))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_keys) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_keys)} -> {len(idxs)})."
                )
            audio_keys = [audio_keys[idx] for idx in idxs]
            mel_keys = [mel_keys[idx] for idx in idxs]

        # assert the number of files
        assert len(audio_keys) == len(
            mel_keys
        ), f"Number of audio and mel files are different ({len(audio_keys)} vs {len(mel_keys)})."

        self.audio_loader = audio_loader
        self.mel_loader = mel_loader
        self.utt_ids = audio_keys
        self.return_utt_id = return_utt_id
        self.return_sampling_rate = return_sampling_rate
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray or tuple: Audio signal (T,) or (w/ sampling rate if return_sampling_rate = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        fs, audio = self.audio_loader[utt_id]
        mel = self.mel_loader[utt_id]

        # normalize audio signal to be [-1, 1]
        audio = audio.astype(np.float32)
        audio /= 1 << (16 - 1)  # assume that wav is PCM 16 bit

        if self.return_sampling_rate:
            audio = (audio, fs)

        if self.return_utt_id:
            items = utt_id, audio, mel
        else:
            items = audio, mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)


class AudioSCPDataset(Dataset):
    """PyTorch compatible audio dataset based on kaldi-stype scp files."""

    def __init__(
        self,
        wav_scp,
        segments=None,
        audio_length_threshold=None,
        return_utt_id=False,
        return_sampling_rate=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            wav_scp (str): Kaldi-style wav.scp file.
            segments (str): Kaldi-style segments file.
            audio_length_threshold (int): Threshold to remove short audio files.
            return_utt_id (bool): Whether to return utterance id.
            return_sampling_rate (bool): Wheter to return sampling rate.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load scp as lazy dict
        audio_loader = kaldiio.load_scp(wav_scp, segments=segments)
        audio_keys = list(audio_loader.keys())

        # filter by threshold
        if audio_length_threshold is not None:
            audio_lengths = [audio.shape[0] for _, audio in audio_loader.values()]
            idxs = [
                idx
                for idx in range(len(audio_keys))
                if audio_lengths[idx] > audio_length_threshold
            ]
            if len(audio_keys) != len(idxs):
                logging.warning(
                    f"Some files are filtered by audio length threshold "
                    f"({len(audio_keys)} -> {len(idxs)})."
                )
            audio_keys = [audio_keys[idx] for idx in idxs]

        self.audio_loader = audio_loader
        self.utt_ids = audio_keys
        self.return_utt_id = return_utt_id
        self.return_sampling_rate = return_sampling_rate
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray or tuple: Audio signal (T,) or (w/ sampling rate if return_sampling_rate = True).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        fs, audio = self.audio_loader[utt_id]

        # normalize audio signal to be [-1, 1]
        audio = audio.astype(np.float32)
        audio /= 1 << (16 - 1)  # assume that wav is PCM 16 bit

        if self.return_sampling_rate:
            audio = (audio, fs)

        if self.return_utt_id:
            items = utt_id, audio
        else:
            items = audio

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)


class MelSCPDataset(Dataset):
    """PyTorch compatible mel dataset based on kaldi-stype scp files."""

    def __init__(
        self,
        feats_scp,
        mel_length_threshold=None,
        return_utt_id=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            feats_scp (str): Kaldi-style fests.scp file.
            mel_length_threshold (int): Threshold to remove short feature files.
            return_utt_id (bool): Whether to return utterance id.
            allow_cache (bool): Whether to allow cache of the loaded files.

        """
        # load scp as lazy dict
        mel_loader = _get_feats_scp_loader(feats_scp)
        mel_keys = list(mel_loader.keys())

        # filter by threshold
        if mel_length_threshold is not None:
            mel_lengths = [mel.shape[0] for mel in mel_loader.values()]
            idxs = [
                idx
                for idx in range(len(mel_keys))
                if mel_lengths[idx] > mel_length_threshold
            ]
            if len(mel_keys) != len(idxs):
                logging.warning(
                    f"Some files are filtered by mel length threshold "
                    f"({len(mel_keys)} -> {len(idxs)})."
                )
            mel_keys = [mel_keys[idx] for idx in idxs]

        self.mel_loader = mel_loader
        self.utt_ids = mel_keys
        self.return_utt_id = return_utt_id
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.utt_ids))]

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_utt_id = True).
            ndarray: Feature (T', C).

        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]

        utt_id = self.utt_ids[idx]
        mel = self.mel_loader[utt_id]

        if self.return_utt_id:
            items = utt_id, mel
        else:
            items = mel

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.utt_ids)
