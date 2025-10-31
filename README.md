# Spectrum Repository <img width="50" height="40" alt="spectrum2" src="https://github.com/user-attachments/assets/2af31d90-d60e-4611-9caf-5cbddd1acafa" style="vertical-align: -100px;" />

Companion repository for [*Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability*](https://arxiv.org/abs/2510.06084)

**Paper: https://arxiv.org/abs/2510.06084**

**Models 🤗: https://huggingface.co/collections/tsor13/spectrum-68dac670f618224845c0fb7d**

**Spectrum Suite** <img width="20" height="20" alt="spectrumsuitelogo" src="https://github.com/user-attachments/assets/f5568f9c-84fa-46e1-86cb-ee73fd1c0a84" style="vertical-align: middle;" /> is a large-scale resource compiled from >40 data sources spanning >90 tasks requiring models to steer to and match diverse distributions ranging from varied human preferences to numerical distributions and more. For illustrative examples, please references [this pdf](https://tsor13.github.io/files/spectrumprompts.pdf). To use it for training or evaluation, follow the instructions in this repo.

**Spectrum Tuning** <img width="20" height="20" alt="spectrum2" src="https://github.com/user-attachments/assets/2af31d90-d60e-4611-9caf-5cbddd1acafa" style="vertical-align: middle;" /> is a simple post-training method utilizing Spectrum Suite to teach models to span and steer to distributions described in either natural language, with examples, or both. Training code can be found in this repo, and trained models on huggingface.

## Setup
First, clone the repository. Then,
```
cd spectrum
```

We use uv for package management. To initialize the environment, run
```
uv sync
uv pip install -e .
```

To download the data, run
```
cd data
bash download.sh
cd -
```
(Note: Some data sources require no download and are accessed directly from huggingface, while others require manual download. For reference, see `data/README.md`)

## In-Context Learning / Steerability Experiments
The main code for running the in-context steerability experiments can be found in `src/spectrum/icl_classes/eval_icl.py`.

Example commands for replicating the gemma-3-12b experiments on the flight task:
```
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --model_name tsor13/spectrum-gemma-3-12b-v0 --format spectrum --dataset flight
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --model_name google/gemma-3-12b-pt --format colon --dataset flight
uv run src/spectrum/icl_classes/eval_icl.py --auto_fixed_examples --batch_size 1 --max_eval_examples 1000 --model_name google/gemma-3-12b-it --format chat --dataset flight
```

A config to generate all commands used for the experiments can be found in `launch_configs/icl_pt_it.yaml` (all of spectrum suite) and `launch_configs/icl_spectrum.yaml` (just the test set of spectrum suite). To print all commands:
```
uv run launch_configs/launch_sbatch_grid.py launch_configs/icl_it_pt.yaml --print
uv run launch_configs/launch_sbatch_grid.py launch_configs/icl_spectrum.yaml --print
```

The accuracy/loss is logged to wandb as `eval/all_acc_geq_1`/`eval/loss_geq_1`.

## Diversity vs. Validity Experiments
The main code for running the diversity / validity experiment can be found in `src/spectrum/diverse_valid/eval_diverse_valid_all.py`.

Example commands for replicating the gemma-3-12b experiments with just a description as input, and logging to wandb:
```
uv run src/spectrum/diverse_valid/eval_diverse_valid_all.py --num_generations 100 --wandb_project diverse_valid_all --prompt_components description --model_name tsor13/spectrum-gemma-3-12b-v0 --template spectrum
uv run src/spectrum/diverse_valid/eval_diverse_valid_all.py --num_generations 100 --wandb_project diverse_valid_all --prompt_components description --model_name google/gemma-3-12b-pt --template colon
uv run src/spectrum/diverse_valid/eval_diverse_valid_all.py --num_generations 100 --wandb_project diverse_valid_all --prompt_components description --model_name google/gemma-3-12b-it --template chat
```

A config to generate all commands used for the experiments can be found in `launch_configs/diverse_valid_all.yaml`. To print all commands:
```
uv run launch_configs/launch_sbatch_grid.py launch_configs/diverse_valid_all.yaml --print
```

The % valid is logged to `aggregate/percent_valid_mean`, the diversity measure to `aggregate/unique_gens_pct_mean`, and the yield to `aggregate/unique_valid_count_mean`.

## Distributional Alignment Experiments
The main code for running the distributional alignment experiments can be found in `src/spectrum/distributional_alignment/eval_distributional.py`.

Example commands for replicating the gemma-3-12b experiments on the urn task:
```
uv run src/spectrum/distributional_alignment/eval_distributional.py --log_wandb --batch_size 16 --max_eval 1000 --random_seed 42 --task urn --model_name tsor13/spectrum-gemma-3-12b-v0 --format spectrum
uv run src/spectrum/distributional_alignment/eval_distributional.py --log_wandb --batch_size 16 --max_eval 1000 --random_seed 42 --task urn --model_name google/gemma-3-12b-pt --format colon
uv run src/spectrum/distributional_alignment/eval_distributional.py --log_wandb --batch_size 16 --max_eval 1000 --random_seed 42 --task urn --model_name google/gemma-3-12b-it --format chat 
```

A config to generate all commands used for the experiments can be found in `launch_configs/distributional.yaml`. To print all commands:
```
uv run launch_configs/launch_sbatch_grid.py launch_configs/distributional.yaml --print
```

The JS-divergences are logged in wandb as `js_divergence.mean`.

## Training
The main code for training can be found in: `src/spectrum/train.py`

An example training run command on a gemma-1b model is:
```
uv run src/spectrum/train.py --model_name google/gemma-3-1b-pt --format spectrum
```
By default, the training script assumes that you have all datasets hydrated. If you have any datasets missing, however, you can tell the training script to ignore the missing datasets with the following flag:
```
uv run src/spectrum/train.py --model_name google/gemma-3-1b-pt --format spectrum --ignore_missing_datasets
```

A config to generate all commands to train the v0 models can be found in `launch_configs/train_v0.yaml`, and a similar script for the v1 models in `launch_configs/train_v1.yaml`. For the main training run, we use accelerate/transformers and utilize 4 80GB A100s.

```
uv run launch_configs/launch_sbatch_grid.py launch_configs/train_v0.yaml --print
uv run launch_configs/launch_sbatch_grid.py launch_configs/train_v1.yaml --print
```




## Citation
```bibtex
@misc{sorensen2025spectrumtuningposttrainingdistributional,
      title={Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability}, 
      author={Taylor Sorensen and Benjamin Newman and Jared Moore and Chan Park and Jillian Fisher and Niloofar Mireshghallah and Liwei Jiang and Yejin Choi},
      year={2025},
      eprint={2510.06084},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.06084}, 
}
```
