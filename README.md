<p align="center">
  <h1 align="center">R2F: Repurposing Ray Frontiers for LLM-free Object Navigation</h1>
  <p align="center">
    <a href="https://fra-tsuna.github.io/website/">Francesco&nbsp;Argenziano</a>
    ·
    <a href="https://www.linkedin.com/in/john-marcelo-a440b62bb?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app">John&nbsp;Mark Alexis Marcelo</a>
    ·
    <a href="https://michelebri.github.io/">Michele&nbsp;Brienza</a>
    ·
    <a href="https://scholar.google.com/citations?user=gfla3a4AAAAJ&hl=it&oi=ao">Abdel&nbsp;Hakim Drid</a>
    ·
    <a href="https://linktr.ee/emanuelemusumeci">Emanuele&nbsp;Musumeci</a>
    ·
    <a href="https://scholar.google.com/citations?user=xZwripcAAAAJ&hl=it&oi=ao">Daniele&nbsp;Nardi</a>
    ·
    <a href="https://scholar.google.com/citations?user=_90LQXQAAAAJ&hl=it&oi=ao">Domenico&nbsp;D. Bloisi</a>
    ·
    <a href="https://scholar.google.com/citations?hl=it&user=Y8LuLfoAAAAJ&view_op=list_works&sortby=pubdate">Vincenzo&nbsp;Suriani</a>
  </p>
  
  <div align="center">

  [![flat](https://img.shields.io/badge/Website-SOON-blue)]() 
  [![arxiv paper](https://img.shields.io/badge/arXiv-SOON-red)]()
  [![license](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
  ![flat](https://img.shields.io/badge/python-3.9-green)
  ![flat](https://img.shields.io/badge/Ubuntu-22.04-E95420)
  ![flat](https://img.shields.io/badge/Ubuntu-24.04-E95420)
  </div>

  </h2>
  

## Setup

```bash
conda create -n r2f python=3.9 cmake=3.22 -y && conda activate r2f 
conda install habitat-sim=0.3.0 -c conda-forge -c aihabitathabitat 
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # required for --vln
```

Model weights for `radio_v2.5-b` and `siglip` are downloaded automatically to `ckpt/` on first run. `HF_HOME` is set to `ckpt/` automatically via Hydra — to relocate the cache, change `hydra.job.env_set.HF_HOME` in [config/config.yaml](config/config.yaml).

## Data

The evaluation tasks can be found under `data/hm3d/val/`.
Follow the instruction from the official [Habitat-Matterport3D](https://aihabitat.org/datasets/hm3d-semantics/) repository to download the following scenes: `813, 824, 827, 829, 848, 853, 871, 876, 880, 894`.
Organize the downloaded data as follows:

```
data/hm3d/val/
  tasks-objnav.csv     # object-nav tasks
  tasks-vln.csv        # VLN tasks with instruction text
  <scene_hash>/
    <scene>.basis.glb  # 3D scene
    <scene>.navmesh    # for the path planner
    <scene>.json.gz    # for annotated gt
```

## Running a single task

```bash
# Object-nav, task 5, with viewer
python run_tasks.py episodes=5

# Headless
python run_tasks.py episodes=5 no_viewer=true

# VLN mode (NLP instruction parsing is automatic)
python run_tasks.py episodes=5 vln=true no_viewer=true

# Save RGB + similarity heatmap frames
python run_tasks.py episodes=5 dump=true no_viewer=true
```

## Batch run

```bash
# All 60 obj-nav tasks, headless
python run_tasks.py episodes=all no_viewer=true

# Subset by range or list
python run_tasks.py episodes=0-9 no_viewer=true
python run_tasks.py 'episodes=0,5,18' no_viewer=true

# Resume an interrupted run
python run_tasks.py episodes=all no_viewer=true resume=true
```


## Key flags
List of all the flags:

Config is in [conf/config.yaml](conf/config.yaml) and can be overridden from the CLI with `key=value` syntax.

| Key | Default | Description |
|---|---|---|
| `episodes` | required | `all`, `0-9`, or `0,5,18` |
| `max_steps` | 1000 | Step budget per episode |
| `map_every` | 5 | Frontier map update interval |
| `no_viewer` | false | Headless mode |
| `vln` | false | Read `tasks-vln.csv`, use instruction text as query (NLP parsing automatic) |
| `dump` | false | Save RGB + similarity heatmap every 5 steps |
| `resume` | false | Skip tasks already in `results.csv` |
| `seed` | 42 | Random seed |


Results are written incrementally to `results/<timestamp>/results.csv`.

## Evaluation

```bash
python eval.py                                  # latest results run
python eval.py csv_path=results/<timestamp>/    # specific run
python eval.py json=true                        # machine-readable output
```

Metrics reported:

| Metric | Description |
|---|---|
| `success_rate` | Fraction of episodes where agent ended within 1.5m of the target |
| `spl` | Success weighted by path length (Anderson et al., 2018) |
| `avg_elapsed_s_on_success` | Mean wall-clock time on successful episodes |
