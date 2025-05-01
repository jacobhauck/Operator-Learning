"""
General utilities for handling external datasets.
"""
import hashlib
import os

import requests
import tqdm


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


def file_size(url):
    """
    Checks how large a file is at the given URL.
    :param url: The file URL
    :return: How many bytes the file occupies
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        assert 'Content-Length' in response.headers, 'Cannot find file size in headers'

        return int(response.headers['Content-Length'])


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

