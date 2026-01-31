uv run ./src/train.py \
  environment=acrobot \
  agent=dqn \
  agent.buffer_size=5000 \
  network=mlp \
  optimizer=adam \
  training.num_episodes=500 \
  wandb.enabled=false \
  rendering.enabled=true
