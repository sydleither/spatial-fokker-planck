# spatial-fokker-planck
Measuring games in spatial data by fitting the Fokker-Planck equation.

## Fit to self
```
python3 -m self_fitting.aggregate 100 0.001
python3 -m self_fitting.aggregate 100 0.01
python3 -m self_fitting.aggregate 500 0.001
python3 -m self_fitting.aggregate 500 0.01
```

## Fit to ABM

### Generate data
```
python3 EGT_HAL/create_sbatch_job.py {email} abm 0-00:05 1gb {path}/spatial-fokker-planck/EGT_HAL {node}
python3 -m in_silico_fitting.generate_abm_data in_silico 5 5 "sbatch job_abm.sb"
bash data/in_silico/5_5/raw/run0.sh
```

### Assess data
```
python3 -m in_silico_fitting.validate_data in_silico 5_5
python3 -m spatial_subsample_test in_silico 5_5 5
```