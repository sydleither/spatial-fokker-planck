# spatial-fokker-planck
Measuring games in spatial data by fitting the Fokker-Planck equation.

## Fit to self
```
python3 -m self_fitting.aggregate
```

## Fit to ABM

### Generate data
```
python3 EGT_HAL/create_sbatch_job.py {email} abm 0-00:05 1gb {path}/spatial-fokker-planck/EGT_HAL {node}
python3 -m in_silico_fitting.generate_abm_data
bash data/in_silico/5_5/raw/run0.sh
```

### Assess data
```
python3 -m aggregate_curves
python3 -m aggregate_curves -d in_silico -src 5_5 -sub 5
python3 -m in_silico_fitting.validate_data in_silico 5_5
python3 -m spatial_subsample_test
```

### Assess fit
```
python3 -m in_silico_fitting.aggregate -i 10000
```