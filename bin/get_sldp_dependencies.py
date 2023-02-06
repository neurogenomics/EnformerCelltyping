"""
Simple function to download the dependencies for the 
[SLDP tool](https://github.com/yakirr/sldp).

The data is downloaded to the inputted directory (./data_download/sldp)
"""

import os
import os.path
from os import path
import sys
import errno

def get_sldp_dependencies(ref_data_path):
    try:#try deals with if it already exists
        os.makedirs(ref_data_path)
    except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    # background model for control
    cmd1 = 'wget -O '+ref_data_path+'/maf5.tar.gz'+' https://data.broadinstitute.org/alkesgroup/SLDP/background/maf5.tar.gz'
    cmd2 = 'tar -xzf '+ref_data_path+'/maf5.tar.gz -C '+ref_data_path+' && rm '+ref_data_path+'/maf5.tar.gz'
    print("Fetching SLDP background model for control")
    os.system(cmd1)
    os.system(cmd2)
    #LD blocks
    cmd1 = 'wget -O '+ref_data_path+'/pickrell_ldblocks.hg19.eur.bed'+' https://data.broadinstitute.org/alkesgroup/SLDP/refpanel/pickrell_ldblocks.hg19.eur.bed'
    print("Fetching SLDP LD Blocks")
    os.system(cmd1)
    #SVDs of LD blocks
    cmd1 = 'wget -O '+ref_data_path+'/svds_95percent.tar'+' https://data.broadinstitute.org/alkesgroup/SLDP/refpanel/svds_95percent.tar'
    cmd2 = 'tar -xf '+ref_data_path+'/svds_95percent.tar -C '+ref_data_path+' && rm '+ref_data_path+'/svds_95percent.tar'
    print("Fetching SLDP SVDs of LD blocks")
    os.system(cmd1)
    os.system(cmd2) 
    #SNP metadata   
    cmd1 = 'wget -O '+ref_data_path+'/KG3.hm3.tar.gz'+' https://data.broadinstitute.org/alkesgroup/SLDP/refpanel/KG3.hm3.tar.gz'
    cmd2 = 'tar -xzf '+ref_data_path+'/KG3.hm3.tar.gz -C '+ref_data_path+' && rm '+ref_data_path+'/KG3.hm3.tar.gz'
    cmd3 = 'wget -O '+ref_data_path+'/KG3.tar.gz'+' https://data.broadinstitute.org/alkesgroup/SLDP/refpanel/KG3.tar.gz'
    cmd4 = 'tar -xzf '+ref_data_path+'/KG3.tar.gz && rm '+ref_data_path+'/KG3.tar.gz'
    print("Fetching SLDP SNP metadata")
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
    os.system(cmd4)
    #list of rsids
    cmd1 = 'wget -O '+ref_data_path+'/1000G_hm3_noMHC.rsid'+' https://data.broadinstitute.org/alkesgroup/SLDP/refpanel/1000G_hm3_noMHC.rsid'
    print("Fetching SLDP RSIDs")
    os.system(cmd1)
    #LD scores
    cmd1 = 'wget -O '+ref_data_path+'/LDscore.tar.gz'+' https://data.broadinstitute.org/alkesgroup/SLDP/refpanel/LDscore.tar.gz'
    cmd2 = 'tar -xzvf '+ref_data_path+'/LDscore.tar.gz -C '+ref_data_path+' && rm '+ref_data_path+'/LDscore.tar.gz'
    print("Fetching SLDP LD scores")
    os.system(cmd1)
    os.system(cmd2)
    #get GWAS sumstats(Crohn's as an example)
    cmd1 = 'wget -O '+ref_data_path+'/CD.tar.gz'+' https://data.broadinstitute.org/alkesgroup/SLDP/sumstats/complex/CD.tar.gz'
    cmd2 = 'tar -xzf '+ref_data_path+'/CD.tar.gz -C '+ref_data_path+' && rm '+ref_data_path+'/CD.tar.gz'
    print("Fetching SLDP GWAS Sumstats Example")
    os.system(cmd1)
    os.system(cmd2)
    #these two files also appear in working directory too so just remove
    os.system('rm ./1000G_hm3_noMHC.rsid')
    os.system('cp -rf ./plink_files/* '+ref_data_path+'/plink_files/')
    os.system('rm -rf ./plink_files')
