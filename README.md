# Discovering and Achieving Goals via World Models

Official implementation of the [Lexa][website] agent from the paper Discovering and Achieving Goals via World Models.

<img src="https://russellmendonca.github.io/data/lexa-method.gif" width="600">


[website]: https://lexa-agent.github.io/



## Setup

Create the conda environment by running : 

```
conda env create -f environment.yml
```

Clone the [lexa-benchmark][lexa-bench-repo] repo, and modify the python path   
`export PYTHONPATH=<path to lexa-training>/lexa:<path to lexa-benchmark>`  

Export the following variables for rendering  
`export MUJOCO_RENDERER=egl; export MUJOCO_GL=egl`

[lexa-bench-repo]: https://github.com/lexa-agent/lexa-benchmark

## Training

First source the environment : `source activate lexa`

For training, run : 

```
export CUDA_VISIBLE_DEVICES=<gpu_id>  
python train.py --configs defaults <method> --task <task> --logdir <log path>
```
where method can be `lexa_temporal`, `lexa_cosine`, `ddl`, `diayn` or `gcsl`   
Supported tasks are `dmc_walker_walk`, `dmc_quadruped_run`, `robobin`, `kitchen`, `joint`

To view the graphs and gifs during training, run `tensorboard --logdir <log path>`

