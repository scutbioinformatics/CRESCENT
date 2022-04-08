# data description
## clinical: clin.csv
csv file of cancer patient survival data\
data can download by Python package TCGAbiolinks.\
Preliminary processing of data that has been described by the following structure is required.
|     | times  |  f  | days_to_death |
|  ----  | ----  | ----  | ----  |
| TCGA patient id1  | val1 | 0 | val3 |
| TCGA patient id1  | val2 | 1 | val4 |
| ...  | ... | ... | ... | 
## network: net.csv
data can download by [IRefIndex](https://irefindex.vib.be/download/irefindex/data/archive/release_16.0/psi_mitab/MITAB2.6/9606.mitab.05-29-2019.txt.zip),[CPDB](http://cpdb.molgen.mpg.de/download/ConsensusPathDB_human_PPI.gz),[STRING-db](https://stringdb-static.org/download/protein.links.v11.0/9606.protein.links.v11.0.txt.gz)\
Preliminary processing of data that has been described by the following structure is required.\
file of ppi,like...
|     | partner1  | partner2 |  confidence |
|  ----  | ----  | ----  |  ----  |
| 0  | STIM1 | TRPC1 | 0.597177 |
| 1  | ATP2B4 | NOS1 | 0.520836 |
| 3  | ... | ... | ... |
## expression: exp.csv
data can download by [RNAseqDB](https://github.com/mskcc/RNAseqDB)\
Preliminary processing of data that has been described by the following structure is required.\
Gene expression data from TCGA cancer patients
|     | TCGA patient id1  | TCGA patient id2 |  ... |
|  ----  | ----  | ----  | ----  |
| gene1  | val1 | val2 | ... |
| gene2  | val3 | val4 | ... |
| gene3  | ... | ... | ... |

For ppi network preprocessing, please refer to [EMOGI]https://github.com/schulter/EMOGI/tree/master/network_preprocessing
