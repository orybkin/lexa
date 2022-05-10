## Discovering and Achieving Goals via World Models

####  [[Project Website]](https://orybkin.github.io/lexa/) [[Benchmark Code]](https://github.com/orybkin/lexa-benchmark) [[Video (2min)]](https://www.youtube.com/watch?v=LnZj2lZYD3k) [[Oral Talk (13min)]](https://www.youtube.com/watch?v=WWHlQbigQp4) [[Paper]](https://arxiv.org/pdf/2110.09514.pdf)
[Russell Mendonca](https://russellmendonca.github.io/)\*<sup>1</sup>, [Oleh Rybkin](https://www.seas.upenn.edu/~oleh/)\*<sup>2</sup>, [Kostas Daniilidis](http://www.cis.upenn.edu/~kostas/)<sup>2</sup>, [Danijar Hafner](https://danijar.com/)<sup>3,4</sup>, [Deepak Pathak](https://www.cs.cmu.edu/~dpathak/)<sup>1</sup><br/>
(&#42; equal contribution, random order)

<sup>1</sup>Carnegie Mellon University </br> 
<sup>2</sup>University of Pennsylvania </br>
<sup>3</sup>Google Research, Brain Team </br> 
<sup>4</sup>University of Toronto </br> 

<img src="https://russellmendonca.github.io/data/lexa-method.gif" width="600">

Official implementation of the [Lexa](https://orybkin.github.io/lexa/) agent from the paper Discovering and Achieving Goals via World Models.

## Setup

Create the conda environment by running : 

```
conda env create -f environment.yml
```

Clone the [lexa-benchmark](https://github.com/orybkin/lexa-benchmark) repo, and modify the python path   
`export PYTHONPATH=<path to lexa-training>/lexa:<path to lexa-benchmark>`  

Export the following variables for rendering  
`export MUJOCO_RENDERER=egl; export MUJOCO_GL=egl`

**WARNING!** Make sure to use the right python and mujoco version. The robobin environment code is known to break with other versions. Other environments might or might not work.

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


## Bibtex
If you find this code useful, please cite:

```
@misc{lexa2021,
    title={Discovering and Achieving Goals via World Models},
    author={Mendonca, Russell and Rybkin, Oleh and
    Daniilidis, Kostas and Hafner, Danijar and Pathak, Deepak},
    year={2021},
    Booktitle={NeurIPS}
}
```

## Acknowledgements
This code was developed using [Dreamer V2](https://github.com/danijar/dreamerv2) and [Plan2Explore](https://github.com/ramanans1/plan2explore).
