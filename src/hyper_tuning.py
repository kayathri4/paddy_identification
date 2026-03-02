import yaml
import subprocess

# Define the grid you want to search
lrs = [1e-4, 5e-5, 1e-5]
batch_sizes = [2, 4]

for lr in lrs:
    for bs in batch_sizes:
        # 1. Update config.yaml
        with open("config.yaml", 'r') as f:
            conf = yaml.safe_load(f)
        
        conf['training']['learning_rate'] = lr
        conf['training']['batch_size'] = bs
        
        with open("config.yaml", 'w') as f:
            yaml.dump(conf, f)
            
        print(f"🚀 Starting Experiment: LR={lr}, BS={bs}")
        # 2. Run the training script
        subprocess.run(["python", "src/train.py"])