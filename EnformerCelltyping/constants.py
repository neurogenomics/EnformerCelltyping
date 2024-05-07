"""All project wide constants are saved in this module."""
import os
import pathlib
import numpy as np

# Note: constants should be UPPER_CASE
constants_path = pathlib.Path(os.path.realpath(__file__))
SRC_PATH = pathlib.Path(os.path.dirname(constants_path))
PROJECT_PATH = pathlib.Path(os.path.dirname(SRC_PATH))

# Data paths
DATA_PATH = PROJECT_PATH / "data"
METADATA_PATH = PROJECT_PATH / "metadata"
SAVE_PATH = PROJECT_PATH / "model_results"

# Load metadata
#  for cell type samples
# dictionary of cell id and descriptive name
CELLS = {'BSS00015' : 'LUNG EPITHELIAL CARCINOMA - 1',
         'BSS00013' : 'LUNG EPITHELIAL CARCINOMA - 2',
         'BSS00035' : 'MUSCLE EWING SARCOMA',
         'BSS00060' : 'ADRENAL GLAND',
         'BSS00078' : 'ANGULAR GYRUS',
         'BSS00087' : 'ASCENDING AORTA',
         'BSS00089' : 'ASTROCYTE',
         'BSS00146' : 'BREAST EPITHELIUM',
         'BSS00171' : 'CARDIAC MUSCLE DERIV',
         'BSS00173' : 'CAUDATE NUCLEUS - 1',
         'BSS00175' : 'CAUDATE NUCLEUS - 2',
         'BSS00214' : 'CHORIONIC VILLUS - 1',
         'BSS00215' : 'CHORIONIC VILLUS - 2',
         'BSS00216' : 'CHORIONIC VILLUS - 3',
         'BSS00217' : 'CHORIONIC VILLUS - 4',
         'BSS00220' : 'CINGULATE GYRUS',
         'BSS00227' : 'COLON MUCOSA - 1',
         'BSS00228' : 'COLON MUCOSA - 2',
         'BSS00231' : 'CD34 CMP - 1',
         'BSS00232' : 'CD34 CMP - 2',
         'BSS00267' : 'ACUTE LYMPHOBLASTIC LEUKEMIA',
         'BSS00270' : 'DUODENUM MUCOSA',
         'BSS00273' : 'ECTODERMAL DERIV',
         'BSS00287' : 'ENDODERMAL DERIV',
         'BSS00296' : 'UMBILICAL VEIN ENDOTHELIAL CELL',
         'BSS00316' : 'ESOPHAGUS - 1',
         'BSS00318' : 'ESOPHAGUS - 2',
         'BSS00341' : 'LUNG FIBROBLAST STRM',
         'BSS00353' : 'FORESKIN FIBROBLAST',
         'BSS00354' : 'FORESKIN KERATINOCYTE',
         'BSS00368' : 'FORESKIN MELANOCYTE',
         'BSS00379' : 'GASTROCNEMIUS MEDIALIS',
         'BSS00439' : 'LYMPHOBLASTOID CELL LINE',
         'BSS00476' : 'SKIN FIBROBLAST',
         'BSS00478' : 'ESC H1',
         'BSS00484' : 'ESC H9',
         'BSS00492' : 'COLORECTAL ADENOCARCINOMA',
         'BSS00507' : 'HEART LEFT VENTRICLE',
         'BSS00512' : 'HEART LEFT VENTRICLE - 1',
         'BSS00513' : 'HEART LEFT VENTRICLE - 2',
         'BSS00525' : 'HEART RIGHT VENTRICLE',
         'BSS00529' : 'CERVIX ADENOCARCINOMA',
         'BSS00558' : 'HEPATOCELLULAR CARCINOMA',
         'BSS00715' : 'ESC HUES48',
         'BSS00716' : 'ESC HUES6',
         'BSS00717' : 'ESC HUES64',
         'BSS00720' : 'LUNG FIBROBLAST STRM EMB',
         'BSS00731' : 'iPSC DF19',
         'BSS00737' : 'iPSC 18a',
         'BSS00739' : 'iPSC 20b',
         'BSS01065' : 'B CELL LYMPHOMA - KARPAS422',
         'BSS01068' : 'KERATINOCYTE',
         'BSS01119' : 'LARGE INTESTINE',
         'BSS01124' : 'HIPPOCAMPUS - 1',
         'BSS01125' : 'HIPPOCAMPUS - 2',
         'BSS01126' : 'HIPPOCAMPUS - 3',
         'BSS01159' : 'LIVER - 1',
         'BSS01168' : 'LIVER - 2',
         'BSS01190' : 'LUNG',
         'BSS01213' : 'MAMMARY EPITHELIAL CELL',
         'BSS01226' : 'MAMMARY GLAND ADENOCARCINOMA',
         'BSS01272' : 'MIDDLE FRONTAL AREA',
         'BSS01274' : 'MYELOMA',
         'BSS01282' : 'RECTUM MUCOSA - 1',
         'BSS01283' : 'RECTUM MUCOSA - 2',
         'BSS01287' : 'DUODENUM MUSCLE',
         'BSS01319' : 'LEG MUSCLE',
         'BSS01332' : 'TRUNK MUSCLE',
         'BSS01344' : 'MYOTUBE',
         'BSS01371' : 'NEURAL PROGENITOR DERIV',
         'BSS01377' : 'NEUROSPHERE',
         'BSS01390' : 'B CELL LYMPHOMA - OCILY3',
         'BSS01397' : 'OSTEOBLAST',
         'BSS01399' : 'OVARY',
         'BSS01405' : 'PANCREAS DUCT EPITHELIAL CARCINOMA',
         'BSS01406' : 'PANCREAS',
         'BSS01414' : 'PROSTATE ADENOCARCINOMA',
         'BSS01415' : 'LUNG ADENOCARCINOMA',
         'BSS01423' : 'MONONUCLEAR CELL',
         'BSS01431' : 'PLACENTA - 1',
         'BSS01438' : 'PLACENTA - 2',
         'BSS01441' : 'PLACENTA - 3',
         'BSS01446' : 'PLACENTA - 4',
         'BSS01463' : 'PSOAS MUSCLE',
         'BSS01545' : 'SIGMOID COLON - 1',
         'BSS01548' : 'SIGMOID COLON - 2',
         'BSS01562' : 'NEUROBLASTOMA',
         'BSS01574' : 'SKELETAL MUSCLE MYOBLAST',
         'BSS01578' : 'SKELETAL MUSCLE',
         'BSS01588' : 'SMALL INTESTINE GI',
         'BSS01601' : 'SMALL INTESTINE GI EMB',
         'BSS01612' : 'SMOOTH MUSCLE DERIV',
         'BSS01631' : 'SPLEEN',
         'BSS01639' : 'STOMACH - 1',
         'BSS01651' : 'STOMACH - 2',
         'BSS01659' : 'STOMACH MUSCLE',
         'BSS01667' : 'ADIPOSE TISSUE',
         'BSS01676' : 'SUBSTANTIA NIGRA',
         'BSS01712' : 'TEMPORAL LOBE - 1',
         'BSS01714' : 'TEMPORAL LOBE - 2',
         'BSS01814' : 'THORACIC AORTA',
         'BSS01820' : 'THYMUS EMB',
         'BSS01825' : 'THYMUS',
         'BSS01857' : 'TROPHOBLAST DERIV'
        }

SAMPLES=list(CELLS.values())
SAMPLE_NAMES=list(CELLS.keys())

#hg19 chrom lengths
CHROM_LEN =np.array([249250621, 243199373, 198022430, 191154276, 180915260, 171115067,
            159138663, 146364022, 141213431, 135534747, 135006516, 133851895,
            115169878, 107349540, 102531392,  90354753,  81195210,  78077248,
            59128983,  63025520,  48129895,  51304566])
#Model will predict on chromsomes 1-22 (not sex chromosomes)
CHROMOSOMES =np.array(['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8',
              'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15',
              'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22'])

MODEL_REFERENCE_PATH = DATA_PATH / "model_ref/"
DNA_REFERENCE_PATH = DATA_PATH / "dna/"
#pre-generated training data
TRAIN_DATA_PATH = DATA_PATH / "train"

#  load DNA data (sequence data) path
DNA_PATHS = list(DNA_REFERENCE_PATH.glob("*.bigWig"))
#  load Average chromatin accessibilty data path
ATAC_AVG_PATH = MODEL_REFERENCE_PATH / "avg_atac.bigWig"
# y values (histone mark) avg paths
h3k27ac_AVG_PATH = MODEL_REFERENCE_PATH / "avg_h3k27ac.bigWig"
h3k4me1_AVG_PATH = MODEL_REFERENCE_PATH / "avg_h3k4me1.bigWig"
h3k4me3_AVG_PATH = MODEL_REFERENCE_PATH / "avg_h3k4me3.bigWig"
h3k9me3_AVG_PATH = MODEL_REFERENCE_PATH / "avg_h3k9me3.bigWig"
h3k27me3_AVG_PATH = MODEL_REFERENCE_PATH / "avg_h3k27me3.bigWig"
h3k36me3_AVG_PATH = MODEL_REFERENCE_PATH / "avg_h3k36me3.bigWig"

# load blacklist regions
BLACKLIST_PATH = MODEL_REFERENCE_PATH / "encode_blacklist.bigBed"

# Standard data paths
DNA_DATA = {path.stem: path for path in DATA_PATH.glob("dna/*")}
ATAC_DATA = {path.stem: path for path in DATA_PATH.glob("atac/*")}
H3K27AC_DATA = {path.stem: path for path in DATA_PATH.glob("h3k27ac/*")}
H3K4ME1_DATA = {path.stem: path for path in DATA_PATH.glob("h3k4me1/*")}
H3K4ME3_DATA = {path.stem: path for path in DATA_PATH.glob("h3k4me3/*")}
H3K9ME3_DATA = {path.stem: path for path in DATA_PATH.glob("h3k9me3/*")}
H3K27ME3_DATA = {path.stem: path for path in DATA_PATH.glob("h3k27me3/*")}
H3K36ME3_DATA = {path.stem: path for path in DATA_PATH.glob("h3k36me3/*")}

DNA = ["A", "C", "G", "T"]
#DNA input same as Enformer - DNA input window, ~100kbp either side
WINDOW_SIZE_DNA = 196_608
#Local chromatin accessibilty input size, ~100kbp either side
WINDOW_SIZE_LCL_CA = 1562*128
#PanglaoDB 1216 marker genes, 3k around TSS
WINDOW_SIZE_GBL_CA = 1216*3_000
#chaneels in output of enf chopped model
ENF_CHANNELS = 1536
#number of predicted positions
ENF_PRED_POS = 896
#Target BP of prediciton window 
TARGET_BP = 896*128


REF_GEN = "hg19"
ALLOWED_FEATURES = ["A", "C", "G", "T", "chrom_access_embed","h3k27ac", "h3k4me1",
		    "h3k4me3","h3k9me3","h3k27me3","h3k36me3","atac","atac_avg"]
ALLOWED_CELLS = SAMPLES
CHROMOSOME_DATA = {
    chromosome: length for chromosome, length in zip(CHROMOSOMES, CHROM_LEN)
}
HIST_MARKS = ["h3k27ac", "h3k4me1","h3k4me3","h3k9me3","h3k27me3","h3k36me3","atac"]
PRED_HIST_MARKS = ["h3k27ac", "h3k4me1","h3k4me3","h3k9me3","h3k27me3","h3k36me3"]
AVG_DATA_PATH = dict(zip(HIST_MARKS, [h3k27ac_AVG_PATH,h3k4me1_AVG_PATH,h3k4me3_AVG_PATH,
                                      h3k9me3_AVG_PATH,h3k27me3_AVG_PATH,h3k36me3_AVG_PATH,
                                      ATAC_AVG_PATH]))
TO_AVG_DATA = dict(zip(HIST_MARKS, [H3K27AC_DATA,H3K4ME1_DATA,H3K4ME3_DATA,
                                 H3K9ME3_DATA,H3K27ME3_DATA,H3K36ME3_DATA,
                                 ATAC_DATA]))
