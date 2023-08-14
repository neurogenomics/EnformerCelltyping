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
import os
import errno
from typing import List
import pandas as pd
from pyarrow import csv
from functools import partial
import pathlib
from multiprocessing import Pool, cpu_count
from EnformerCelltyping.constants import HIST_MARKS, METADATA_PATH, DATA_PATH, PROJECT_PATH
import tarfile
import zipfile

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

def download_bigwigs(exp_type: str) -> None:
    """
    Download the turing files to their folders
    in the data_download folder. File urls are read from
    the files in metadata

    Parameters
    ----------
    exp_type: string
        the test type, h3k4me1, h3k27ac, atac or dna
    """

    if exp_type not in ['atac','dna','model_ref']+HIST_MARKS:
        raise ValueError

    df = pd.read_csv(METADATA_PATH / '{exp_type}.csv'.format(exp_type=exp_type),
            sep=',')
    #convert to list
    name_urls = df.values.tolist()
    folder_name = DATA_PATH / '{exp_type}'.format(exp_type=exp_type)
    #create parallel use all cpus
    pool = Pool(cpu_count())
    #same folder_name for all
    download_func = partial(download_file, folder = folder_name)
    results = pool.map(download_func, name_urls)
    #close parallel
    pool.close()
    pool.join()

def download_blacklist_regions() -> None:
    """
    Downloads ENCODE's blacklist regions in hg19
    """
    folder_name = DATA_PATH / 'model_ref'
    blck_list = ("encode_blacklist.bigBed",
            "https://www.encodeproject.org/files/ENCFF000KJP/@@download/ENCFF000KJP.bigBed")
    download_file(blck_list,folder_name,extension=False)
    
def download_Enformer_Celltyping_weights() -> None:
    """
    Downloads Enformer Celltyping model weights
    """
    folder_name = PROJECT_PATH / 'EnformerCelltyping'
    enf_cell = ("enformer_celltyping_weights.zip",
            "https://figshare.com/ndownloader/files/39678760")
    download_file(enf_cell,folder_name,extension=False)
    #now unzip
    zip_pth = str(PROJECT_PATH / 'EnformerCelltyping' / 'enformer_celltyping_weights.zip')
    with zipfile.ZipFile(zip_pth, 'r') as zip_ref:
        zip_ref.extractall(str(PROJECT_PATH / 'EnformerCelltyping'))
    #delete tar file
    os.remove(str(PROJECT_PATH / 'EnformerCelltyping' / 'enformer_celltyping_weights.zip'))

def download_avg_chromatin_accessibility() -> None:
    """
    Downloads the average singal of chromatin accessibility
    for the 103 EpiMap, training cell types.
    """
    folder_name = DATA_PATH / 'model_ref'
    avg_atac = ("avg_atac",
                "https://figshare.com/ndownloader/files/39111956")
    download_file(avg_atac,folder_name)  
    
def download_nott_19_chrom_access() -> None:
    """
    Downloads the processed chromatin accessibility data for microglia
    from Nott et al., 2019, averaged at 128 base-pairs, ready for input
    into Enformer Celltyping
    """
    folder_name = DATA_PATH / 'demo'
    microg = ("Nott19_Microglia_128",
                "https://figshare.com/ndownloader/files/39140972")
    download_file(microg,folder_name)     
    
def download_enformer() -> None:
    """
    Downloads the enformer model from Tensorflow Hub
    """
    folder_name = DATA_PATH
    enf_mod = ("enformer_model.tar.gz",
                "https://tfhub.dev/deepmind/enformer/1?tf-hub-format=compressed")
    download_file(enf_mod,folder_name,extension=False)
    #now unzip
    tar = tarfile.open(str(DATA_PATH / 'enformer_model.tar.gz'), 'r:gz')
    # create destination dir if it does not exist
    if os.path.isdir(str(DATA_PATH / 'enformer_model')) == False:
        os.mkdir(str(DATA_PATH / 'enformer_model'))
    tar.extractall(str(DATA_PATH / 'enformer_model'))
    tar.close()
    #delete tar file
    os.remove(str(DATA_PATH / 'enformer_model.tar.gz'))

if __name__ == '__main__':
    print(datetime.now())
    print("There are {} CPUs on this machine ".format(cpu_count()))
    #download DNA bigWigs
    for exp_type in ['dna']:
        print("Downloading {exp_type}".format(exp_type=exp_type))
        download_bigwigs(exp_type)
    #download encode blacklist regions
    download_blacklist_regions()
    #download avg chromatin accessibility bigWigs
    download_avg_chromatin_accessibility()
    #download microglia data
    download_nott_19_chrom_access()
    #download enformer from tensorflow hub
    download_enformer()
    # download Enformer Celltyping weights
    download_Enformer_Celltyping_weights()
    print("All downloads complete")
    print(datetime.now())
