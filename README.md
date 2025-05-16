# spatial-fokker-planck
Measuring games in spatial data by fitting the Fokker-Planck equation.

## Fit to ABM

### Generate data
```
python3 EGT_HAL/create_sbatch_job.py {email} abm 0-00:05 1gb {path}/spatial-fokker-planck/EGT_HAL {node}
python3 generate_abm_data.py in_silico HAL 100 "sbatch job_abm.sb"
```