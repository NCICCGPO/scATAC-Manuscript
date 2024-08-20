# ISM scoring for motifs and mutations

## motif ISM

#### scoring
```
python -u ./scripts/motifism_score.py --file_specifier brca --program_switch main --file none
```

#### aggregate results
```
python -u ./scripts/motifism_aggregate.py --filepath ./ism_cleaning_output/brca/
```

## mutation-context ISM

### scoring

```
python -u ./mutationism_score.py --file_specifier brca --program_switch main --file <path to variants file>
```

### aggregate results

```
python ./mutationism_aggregate.py --output_dir <output from previous command> --export results.csv
```

