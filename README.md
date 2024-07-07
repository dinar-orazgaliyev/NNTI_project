# NNTI Project WS 2023/2024

Members:
- Filippo Garosi 7042750 figa00001@stud.uni-saarland.de
- Dinar Orazgaliyiev 7056420 dior@stud.uni-saarland.de

## About

This is the repository for the end-of-semester NNTI 23/24 Project. See [tasks](./tasks) for a list of implemented functionalities.

## Notes

The code is ready to be executed locally and on the Saarland University compute cluster, depending on the task.

## Setup
Before executing code locally, use `pip install -f requirements.txt`. For cluster usage, be sure to run `condor_submit setup.sub`.

# Running tasks / implementation

- Task 1 can be run locally using the Jupyter Notebook [Task1](./notebooks/task1.ipynb).

- Task 2.1 (saving hidden representations to disk) is implemented in [scripts/task2.py](./scripts/task2.py). A different run(something).sh script was specified, per different model, finetuned or not. For the original model, see [run2.sh](./submit_files/run2.sh). It can be run locally by calling the previous script, or scheduled to be run on the cluster using `condor_submit` [submit_files/task2.sub](./submit_files/task2.sub). 

- Task 2.2 (plotting hidden representations) is implemented in [scripts/task2_plot_saving.py](./scripts/task2_plot_saving.py). Similarly, it can be executed locally or on the compute cluster. There are as many run and submit scripts as the models (default and finetuned with {full, bitfit, LoRA, IA^3}). For the default model check out [run2_plot.sh](./submit_files/run2_plot.sh) and [submit_files/task2_plot.sub](./submit_files/task2_plot.sub).

- Task 3 (finetuning) is implemented in [scripts/task3.py](./scripts/task3.py). We recommend executing fine-tuning on the cluster, using 
submit_files/task3_{full, bitfit, lora, ia3}.sub (for example, [submit_files/task3_full.sub](./submit_files/task3_full.sub)).

## Arguments
[scripts/task2.py](./scripts/task2.py), [scripts/task2_plot_saving.py](./scripts/task2_plot_saving.py) and [scripts/task3.py](./scripts/task3.py) have many command-line arguments in order to customize filenames, destination folders, training arguments and more. You can see which options are available by calling the script with no arguments. Here we will mention the least intuitive options: 

- [task2.py](./scripts/task2.py):
    - `--subset-size n` specify to plot tokens and sentences from `n` randomly sampled sentences from the [Flores](https://huggingface.co/datasets/facebook/flores) dataset.
    - `-f FILENAME` specify where to save hidden representations
    - `--model-path` specifies the model to use: it can also be a local (in this case, finetuned) model.

-  [task2_plot_saving.py](./scripts/task2_plot_saving.py):
    - `-f FILENAME` specify where to load hidden representations
    - `--save-dir SAVE_DIR` specify the folder to save hidden space plots in
    - `--average-token-duplicates` flag that, if set, average token representations if multiple exist for the same token. Useful for plots studying spaces of semantics of tokens.


- [task3.py](./scripts/task3.py):
    - `--checkpoint-dir CHECKPOINT_DIR` dir to save checkpoints in
    - `--resume-from RESUME_FROM` path to a model checkpoint
    - `--train-ds / --test-ds` specify which dataset to use for training and testing. Only allows select choices, for which splits, configs etc. are already defined in [task3.py](./scripts/task3.py) itself. For example, `flores` defines a dictionary of datasets for a series of considered languages, and is the default for `--test-ds`.
    - `--warmup-steps WARMUP_STEPS` warmup steps for a linear learning rate scheduler with warmup (read more [here](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup))
    - `--wandb-key WANDB_KEY`: api key for the wandb.ai platform. [task3.py](./scripts/task3.py) send all logs regarding training to the platform, for efficient monitoring, troubleshooting, etc. It automatically provides graphs that plot quantities coming from the logs, over time.