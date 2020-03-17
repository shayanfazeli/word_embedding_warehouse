__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'
__credit__ = 'erLab - University of California, Los Angeles'

# libraries
import os, sys
import requests


def download_file_to_path(file_link: str, path: str, output_file_name: str = None) -> None:
    """
    The file downloader method, `download_file_to_path`, (which is inspired by a stackoverflow post which unfortunately
    I do not recall the address, and could not find it again). It also includes the progress bar and other 
    functionalities.
    
    Parameters
    ----------
    file_link: `str`, required
        The file download link is set in this parameter.
    path: `str`, required
        The folder for the file to be downloaded to.
    output_file_name: `str`, optional (default=None)
        This is the output file name which enforces renaming if specified.
    """
    assert not os.path.isfile(path), "please insert a path to a folder, not a file"
    if not os.path.isdir(path):
        os.makedirs(path)

    file_name = file_link.split('/')[-1]
    link = file_link

    output_file_path = os.path.join(path, file_name) if output_file_name is None else os.path.join(path,
                                                                                                   output_file_name)
    with open(output_file_path, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()
