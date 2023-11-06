"""
Selection of functions to download data from sources like EpiMap and
Blueprint.


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
import get_sldp_dependencies
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


def convert_to_parquet(gz_csv_file: str, sep: str, target_folder: str) -> None:
    """
    This function converts g-zipped csv files to the more compact
    parquet format.

    Parameters
    ----------
    gz_csv_file: str
        the path to the gzipped csv file
    sep: str
        the separator in the csv file e.g. ';' or ','
    target_folder: str
        the folder in wich to write the parquet file
    """
    parquet_file = gz_csv_file.split('.')[-3] + '.parquet'
    parquet_file = parquet_file.split('/')[-1]
    parquet_file = './' + target_folder + parquet_file
    create_path(parquet_file)
    table = csv.read_csv(gz_csv_file,
                         parse_options=csv.ParseOptions(delimiter=sep))
    parquet.write_table(table, parquet_file)
    del table
    return parquet_file


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


def download_qtl():
    """
    This funciton downloads the qtl files
    and returns the name of the files.
    """
    cells = ["mono", "neut", "tcel"]
    mods = ["K27AC", "K4ME1"]
    base_link = ("http://ftp.ebi.ac.uk/pub/databases/blueprint/"
                 "blueprint_Epivar/qtl_as/QTL_RESULTS/{cell}_{mod}"
                 "_log2rpm_peer_10_all_summary.txt.gz"
                 )
    name_list = ["{cell}_{mod}.txt.gz".format(cell=cell,mod=mod)
                 for mod in mods for cell in cells]
    url_list = [base_link.format(cell=cell, mod=mod)
                for mod in mods for cell in cells]
    #make tuple of name and link
    name_urls = zip(name_list, url_list)
    #same folder for all QTLs
    folder_name = pathlib.Path('./data/qtl_tmp')
    #create parallel use all cpus
    pool = Pool(cpu_count())
    #same folder_name for all
    download_func = partial(download_file, folder = folder_name)
    results = pool.map(download_func, name_urls)
    #close parallel
    pool.close()
    pool.join()
    return(name_list)


def qtl() -> None:
    """
    This function downloads the qtl files
    and then converts them to parquet files
    """
    txt_files = download_qtl()


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

def download_avg_chromatin_accessibility() -> None:
    """
    Downloads the average singal of chromatin accessibility
    for the 103 EpiMap, training cell types.
    """
    folder_name = DATA_PATH / 'model_ref'
    avg_atac = ("avg_atac",
                "https://figshare.com/ndownloader/files/39111956")
    download_file(avg_atac,folder_name)      

if __name__ == '__main__':
    print(datetime.now())
    print("There are {} CPUs on this machine ".format(cpu_count()))
    for exp_type in ['model_ref','dna']+HIST_MARKS:
        print("Downloading {exp_type}".format(exp_type=exp_type))
        download_bigwigs(exp_type)
    print("download training data completed")
    print(datetime.now())
    #download QTL data
    qtl()
    #download SLDP data
    ref_data_path = "./data/sldp"
    get_sldp_dependencies.get_sldp_dependencies(ref_data_path)
    #download encode blacklist regions
    download_blacklist_regions()
    #download enformer from tensorflow hub
    download_enformer()
    #download avg chromatin accessibility bigWigs
    download_avg_chromatin_accessibility()
    print("All downloads complete")
    print(datetime.now())
