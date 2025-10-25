<h1 align="center">G.O.D Subnet</h1>


🚀 Welcome to the [Gradients on Demand](https://gradients.io) Subnet

> Distributed intelligence for LLM and diffusion model training. Where the world's best AutoML minds compete.

 **Tournaments** 🏆
Competitive events where validators execute miners' open-source training scripts on dedicated infrastructure.

- **Duration**: 4-7 days per tournament
- **Frequency**: New tournaments start 24 hours after the previous one ends
- **Rewards**: Significantly higher weight potential for top performers
- **Open Source**: Winning AutoML scripts are released when tournaments complete
- **Winners Repository**: First and second place tournament scripts are uploaded to [github.com/gradients-opensource](https://github.com/gradients-opensource) 🤙
- [Tournament Overview](docs/tournament_overview.md)
- [Tournament Miner Guide](docs/tourn_miner.md)

## Setup Guides

- [Tournament Miner Guide](docs/tourn_miner.md)
- [Validator Setup Guide](docs/validator_setup.md)

## Developer Resources

For technical documentation on GRPO reward functions and implementation details, see [GRPO Safe Code Execution Guide](docs/grpo_safe_execution.md).

## Recommended Compute Requirements

[Compute Requirements](docs/compute.md)

## Miner Advice

[Miner Advice](docs/miner_advice.md)



## Running evaluations on your own
You can re-evaluate existing tasks on your own machine. Or you can run non-submitted models to check if they are good. 
This works for tasks not older than 7 days.

Make sure to build the latest docker images before running the evaluation.
```bash
docker build -f dockerfiles/validator.dockerfile -t weightswandering/tuning_vali:latest .
docker build -f dockerfiles/validator-diffusion.dockerfile -t diagonalge/tuning_validator_diffusion:latest .
```

To see the available options, run:
```bash
python -m utils.run_evaluation --help
```

To re-evaluate a task, run:
```bash
python -m utils.run_evaluation --task_id <task_id>
```

To run a non-submitted model, run:
```bash
python -m utils.run_evaluation --task_id <task_id> --models <model_name>
```

run my tournament code, run:
```bash
git clone my-repo

curl -Ls https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
cd turn-01
sudo -E ./bootstrap.sh
source $HOME/.bashrc
source $HOME/.venv/bin/activate
uv pip install -e .

# Text task examples
./examples/run_chat_task.sh
./examples/run_instruct_task.sh
./examples/run_dpo_task.sh
./examples/run_grpo_task.sh

# Image task examples
./examples/run_image_task.sh
./examples/run_image_flux_task.sh
```