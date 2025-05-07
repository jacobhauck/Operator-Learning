import json
import os
import shutil
import zipfile

import huggingface_hub as hf
import numpy as np
import torch.utils.data
from huggingface_hub import hf_hub_download


CACHE_PATH = '.cfdbench.cache'
HUGGINGFACE_REPO_ID = 'chen-yingfa/CFDBench'
CATEGORIES = {
    'cylinder',
    'cavity',
    'dam',
    'tube'
}
DATASETS = {
    'bc',
    'geo',
    'prop'
}


def _validate_cache(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    cache_ignore_path = os.path.join(CACHE_PATH, '.gitignore')
    if not os.path.exists(cache_ignore_path):
        with open(cache_ignore_path, 'w') as f:
            f.write('*')


class CFDBenchDataset(torch.utils.data.Dataset):
    def __init__(self, name):
        self.name = name
        self.category, self.dataset = name.split('-')

        assert self.category in CATEGORIES, 'Invalid category'
        assert self.dataset in DATASETS, 'Invalid dataset'

        self.folder_path = str(os.path.join(CACHE_PATH, self.category, self.dataset))
        self._cases = None

    def download(self, overwrite=True):
        if not overwrite and os.path.exists(self.folder_path):
            return

        _validate_cache(self.folder_path)

        if self.category == 'cylinder':
            print('Downloading dataset...')
            downloaded_file = hf_hub_download(
                repo_id=HUGGINGFACE_REPO_ID,
                filename=f'cylinder/{self.dataset}.zip',
                repo_type='dataset',
                local_dir='.cfdbench.cache'
            )

            print('Extracting data...')
            with zipfile.ZipFile(downloaded_file) as f:
                f.extractall(os.path.join(CACHE_PATH, self.category, self.dataset))

            print('Cleaning up...')
            os.remove(downloaded_file)
        else:
            print('Downloading category data...')
            downloaded_file = hf.hf_hub_download(
                repo_id=HUGGINGFACE_REPO_ID,
                filename=self.category + '.zip',
                repo_type='dataset',
                local_dir='.cfdbench.cache'
            )

            print('Extracting data...')
            with zipfile.ZipFile(downloaded_file) as f:
                f.extractall(os.path.join(CACHE_PATH, self.category))

            print('Cleaning up...')
            os.remove(downloaded_file)
            if self.category == 'dam':
                os.remove(os.path.join(CACHE_PATH, self.category, 'rename_case_params.py'))

    def clear_cached_files(self):
        shutil.rmtree(self.folder_path)

    @property
    def cases(self):
        if self._cases is None:
            self._cases = os.listdir(self.folder_path)
        return self._cases

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, item):
        folder = os.path.join(self.folder_path, self.cases[item])
        with open(os.path.join(folder, 'case.json')) as f:
            metadata = json.load(f)

        u = torch.from_numpy(np.load(os.path.join(folder, 'u.npy')))
        v = torch.from_numpy(np.load(os.path.join(folder, 'v.npy')))

