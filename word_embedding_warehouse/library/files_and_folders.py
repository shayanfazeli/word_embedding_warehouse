__author__ = 'Shayan Fazeli'
__email__ = 'shayan@cs.ucla.edu'
__credit__ = 'erLab - University of California, Los Angeles'

# libraries
import os


def force_folder_to_exist(folder_path: str, erase_if_exist: bool = False) -> str:
    """
    The :func:`force_folder_to_exist` forces a string folder to exist.

    Parameters
    ----------
    folder_path: `str`, required
        The path to the folder, which might not exist.
    erase_if_exist: `bool`, optional (default=False)
        The variable that decides to erase the folder if it already exists (on macOS and Linux).

    Returns
    ----------
    The absolute path to the requested folder.
    """
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    elif erase_if_exist:
        os.system('rm -rf ' + folder_path)
        os.makedirs(folder_path)
    return folder_path


def keep_or_delete_file(file_path: str) -> None:
    """
    This method receives a file_path and interactively gives the user the option to keep or delete it.

    Parameters
    ----------
    file_path: `str`, required
        The path to the file
    """
    file_path = os.path.abspath(file_path)
    if os.path.isfile(file_path):
        choice = 'y'  # input('The file at \n\t{}\n already exists, should it be deleted?\n'.format(file_path))
        if choice == 'y':
            os.system('rm ' + file_path)


def erase_file_if_exists(file_path: str) -> None:
    """
    The :func:`erase_file_if_exists` method receives a file_path and erases it if it exists.

    Parameters
    ----------
    file_path: `str`, required
        The path to the file
    """
    file_path = os.path.abspath(file_path)
    if os.path.isfile(file_path):
        choice = input('The file at \n\t{}\n already exists, should it be deleted?\n'.format(file_path))
        if choice == 'y':
            os.system('rm ' + file_path)
