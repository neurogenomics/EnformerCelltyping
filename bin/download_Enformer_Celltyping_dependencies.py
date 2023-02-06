"""
Selection of functions to download data necessary to use 
Enformer Celltyping.


MIT License

Copyright (c) 2021

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from datetime import datetime
import requests
import pandas as pd
import os
import errno
from typing import List
from pyarrow import csv, parquet
from functools import partial
import pathlib
from multiprocessing import Pool, cpu_count
from EnformerCelltyping.constants import HIST_MARKS, METADATA_PATH, DATA_PATH

def create_path(filename: str) -> None:
    """
    This function creates the path to the
    specified folder.

    Parameters
    ----------
    filename : str
        the path to be created
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def download_file(url_name: tuple,folder: str, extension : bool = True) -> str:
    """
    This function downloads a file from the specified url
    and writes it to a folder.

    Parameters
    ----------
    url_name: tuple
        the name for and url from which to download the file.
    folder: str
        the folder in which to write the file
    """
    name = url_name[0]
    url = url_name[1]
    print('Downloading: {name}'.format(name=name))
    #don't add bigwig extension for QTL files or if specified
    if folder.stem == "qtl_tmp" or extension == False:
        local_filename = str(folder) + '/' + name
    else:
        local_filename = str(folder) + '/' + name + ".bigWig"
    try:
        create_path(local_filename)
        with requests.get(url, stream=True) as r:
            #check if link didn't work
            if r.status_code == 404:
                print("Something went wrong when attempting to download from {}".format(name))
            else:
                r.raise_for_status()
                with open(local_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)         
    except:
        print("Something went wrong when attempting to download from {}".format(name))
    return local_filename


def print_names(file_list: List[str]) -> str:
    return '\n'.join(['Saved file at: {}'.format(file) for file in file_list])


def download_blacklist_regions() -> None:
    """
    Downloads ENCODE's blacklist regions in hg19
    """
    folder_name = DATA_PATH / 'model_ref'
    blck_list = ("encode_blacklist.bigBed",
            "https://www.encodeproject.org/files/ENCFF000KJP/@@download/ENCFF000KJP.bigBed")
    download_file(blck_list,folder_name,extension=False)

if __name__ == '__main__':
    print(datetime.now())
    print("There are {} CPUs on this machine ".format(cpu_count()))
    #download encode blacklist regions
    download_blacklist_regions()
    #download avg chromatin accessibility bigWigs

    print("All downloads complete")
    print(datetime.now())
