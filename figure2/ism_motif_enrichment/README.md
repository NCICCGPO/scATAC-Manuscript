# Post-hoc inference tools for deep-learning based prioritization of somatic mutations in cancer

# Setup

```
conda env create --name seq2atac --file seq2atac.yml
conda activate seq2atac
```

# Examples

## motif ISM

*in silico* mutagenesis for modles (although peculiar to BPNet for now)

Chunking is a big feature and with 48M motifs, this can be quite taxing on the memory. Greedy storage of these results with 6 shuffles creates results files of nearly 500G

### pipeline

```
python -u ./scripts/motifism_v12.py --file_specifier brca --program_switch main --file none
```

### chk

It is important to chk that all chunks were computed properly, since sometimes GPUs can crash unexpectedly. To do this,

1. run sep python script

```
import sys
sys.path.append('./')
from reg_diffs.scripts import evalism_v22 as evalism
motifs = './vierstra_Archetype_BRCA_cancer_control_consensusagg.csv.bed'
pfp = './results/tmp/'
suffix = '_indbrcavctrl_v122'
samples = ['brca13', 'brca14', 'brca15', 'brca20', 'brca21', 'brca22', 'brca23', 'brca24', 'brca25']
output_files = [os.path.join(pfp, s+suffix) for s in samples]
chk = evalism.chk_motifism(output_files=output_files, n_lines=89456402, chunksize=1e5, ) # feed filename if don't know n motifs
```

2. Call a special instance of the queing script:
```
python -u ./scripts/motifism_v12.py --file_specifier $2 --program_switch $3 --file $4 --rerun "${@:5}"
```

3. Check again, then proceed


### eval

#### agg chunks
```
python -u ./scripts/evalism_v22.py --filepath ./tmp/reg_diffs/results/tmp/brca_v122/
```
#### merge SHAP and ISM scores to create a list of active motifs

```
python -u ./scripts/evalism_v22.py --filepath brstctrl_merge
```

## mutation-context ISM

### pipeline

```
python -u ./scripts/genloc_ism.py --file_specifier brca --program_switch main --file none
```

### eval

```
python ./experiments/eval_glism.py --output_dir=./results/genomewide/ --export=./results/genomewide/all_cancers_all_mutations.csv
```

## fast tomtom query

```
python ./experiments/run_sample_tomtomq.py --samples brca blca --filesuffix v8_gwide
```

## Enrichment

### motifs

See [this notebook](https://git.illumina.com/nravindra/mutation_prioritization_tooling/blob/main/notebooks/motif_enrich_gbmsubclones.ipynb)

### mutations

```
Under code review
```

# Citation

Neal G. Ravindra and Laksshman Sundaram, Illumina AI Research Lab
- Paper forthcoming

