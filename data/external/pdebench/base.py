import os
import json
import math
import hashlib

import torch.utils.data
import pandas
import requests
import tqdm

import operatorlearning.function as olf


CACHE_PATH = '.pdebench.cache'


_database = None

def get_database():
    global _database

    if _database is None:
        _database = _Database()

    return _database


def calc_md5(file_path, chunk_size=None):
    """
    Calculates the MD5 checksum of a file.

    :param file_path: Path to file to check
    :param chunk_size: Size of chunk size to stream the file. None to load
        the entire file in one chunk
    :return: The MD5 checksum of the file
    """
    md5 = hashlib.md5(usedforsecurity=False)

    with open(file_path, 'rb') as f:
        # Read entire file and compute MD5 if chunk_size is None
        if chunk_size is None:
            md5.update(f.read())
        else:
            # Otherwise, read chunks from f and update the MD5 calculator
            # one chunk at a time
            while True:
                chunk = f.read(chunk_size)
                if len(chunk) == 0:
                    break

                md5.update(chunk)

    # Return final MD5 value
    return md5.hexdigest()


def download_file(url, root, file_name=None, chunk_size=1024*1024, md5=None):
    """
    Downloads a file from a given URL using streaming.

    :param url: The URL of the file to download.
    :param root: The directory in which to store the downloaded file
    :param file_name: The name of the downloaded file; uses the name from the
        URL if None is given.
    :param chunk_size: Streaming chunk size. Bigger chunk size means faster
        download speed buy higher memory usage. Default = 1024kB.
    :param md5: MD5 checksum to verify integrity of download (optional)
    """
    # Get file name from URL if not provided
    if file_name is None:
        file_name = url.split('/')[-1]

    output_file = os.path.join(root, file_name)

    # Get file using requests stream
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        # Create progress bar
        length = response.headers.get('Content-Length')
        if length is not None:
            length = int(length)
        pbar = tqdm.tqdm(
            total=length,
            unit='B',
            unit_scale=True
        )

        # Stream file
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))  # Use actual chunk length, not chunk_size
        pbar.close()

    # Check file integrity using MD5 checksum
    if md5 is not None:
        if calc_md5(output_file, chunk_size=1024*1024) != md5:
            raise RuntimeError('Downloaded file was corrupted; please try again.')


def _validate_cache(dir_name):
    os.makedirs(dir_name, exist_ok=True)
    cache_ignore_path = os.path.join(CACHE_PATH, '.gitignore')
    if not os.path.exists(cache_ignore_path):
        with open(cache_ignore_path, 'w') as f:
            f.write('*')


class _DatabaseRecord:
    @staticmethod
    def copy_interval(interval):
        return tuple(tuple(map(float, interval0)) for interval0 in interval)

    def __init__(self, rows):
        main = rows.iloc[0]
        self._pde = main['PDE']
        self._urls = [url for url in rows['URL']]
        self._local_file_paths = [
            os.path.join(CACHE_PATH, path, filename)
            for path, filename in zip(rows['Path'], rows['Filename'])
        ]
        self._md5 = main['MD5']
        self._spatial_shape = tuple(map(int, json.loads(main['Sshape'])))
        self._spatial_interval = self.copy_interval(json.loads(main['Sinterval']))

        if math.isnan(main['Tshape']):
            self._temporal_shape = None
            self._temporal_interval = None
        else:
            self._temporal_shape = int(main['Tshape'])
            self._temporal_interval = tuple(map(float, json.loads(main['Tinterval'])))

        self._params = json.loads(main['params'])

    def download(self, overwrite=False, chunk_size=16*1024*1024):
        file_num = 0
        for url, local_path in zip(self.urls, self.local_file_paths):
            if not overwrite and os.path.exists(local_path):
                continue

            dir_name, file_name = os.path.split(local_path)
            _validate_cache(dir_name)

            print(f'Downloading file: {url}')
            print(f'File {file_num + 1}/{len(self._urls)}')
            download_file(url, dir_name, file_name, chunk_size=chunk_size, md5=self.md5)

            file_num += 1

    def clear_cached_files(self):
        while True:
            answer = input('Really delete cached dataset files? (y/n)')
            if answer.lower() == 'y':
                break
            elif answer.lower() == 'n':
                print('Canceled deleting files.')
                return
            else:
                print('Invalid input.')

        for file in self.local_file_paths:
            os.remove(file)

    def is_downloaded(self):
        return all(os.path.exists(file) for file in self.local_file_paths)

    @property
    def pde(self):
        return self._pde

    @property
    def local_file_paths(self):
        yield from self._local_file_paths

    @property
    def urls(self):
        yield from self._urls

    @property
    def md5(self):
        return self._md5

    @property
    def spatial_shape(self):
        return self._spatial_shape

    @property
    def temporal_shape(self):
        return self._temporal_shape

    @property
    def spatial_interval(self):
        return self._spatial_interval

    @property
    def temporal_interval(self):
        return self._temporal_interval

    @property
    def dimension(self):
        return len(self._spatial_shape)

    @property
    def is_time_dependent(self):
        return self._temporal_shape is not None

    @property
    def params(self):
        return dict(self._params)

    def __repr__(self):
        s = f'{self.pde}-{self.dimension}D'
        for k, v in self.params.items():
            s += f'-{k}={v}'
        return s


class _Database:
    META_FILE_PATH = os.path.join('data', 'external', 'pdebench', 'metadata.csv')

    def __init__(self):
        self.raw_metadata = pandas.read_csv(_Database.META_FILE_PATH)
        self.records = self.raw_metadata.groupby('ID').apply(_DatabaseRecord)

    def find_datasets(self, pde, dimension, **params):
        candidates = []

        for record in self.records:
            if record.pde != pde:
                continue

            if record.dimension != dimension:
                continue

            for k, v in params.items():
                # noinspection PyProtectedMember
                if k not in record._params:
                    break

                # noinspection PyProtectedMember
                if record._params[k] != v:
                    break
            else:
                candidates.append(record)

        return candidates


class PDEBenchStandardDataset(torch.utils.data.Dataset):
    def __init__(self, pde, dimension, **params):
        candidates = get_database().find_datasets(pde, dimension, **params)
        if len(candidates) != 1:
            raise ValueError('Identification parameters did not determine a unique dataset.')

        self.database_record = candidates[0]
