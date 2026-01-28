uv run ./src/train.py \
  environment=cartpole \
  agent=dqn \
  network=mlp \
  optimizer=adam \
  training.num_episodes=500 \
  wandb.enabled=false
