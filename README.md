# tcn-carnatic-tracker
TCN for Beat and Downbeat Tracking of Carnatic Music

Currently in Development...

## Weights & Biases setup

This project supports:
- online logging (requires a personal W&B API key), or
- offline logging by default (no key needed; sync later with `wandb sync`).

To enable online logging:
1. Create a free account at https://wandb.ai
2. Find your API key in Settings → API keys
3. On macOS:
   - Temporary: `export WANDB_API_KEY="your-key"`
   - Or persist: `echo 'export WANDB_API_KEY="your-key"' >> ~/.zshrc && source ~/.zshrc`
4. Run training normally. Use `--disable-wandb` to turn logging off.
