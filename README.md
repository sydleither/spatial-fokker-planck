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
python3 generate_abm_data.py in_silico 5 5 "sbatch job_abm.sb"
```