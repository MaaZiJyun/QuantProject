#!/bin/bash
#SBATCH --job-name=master_model
#SBATCH --gres=gpu:4g.40gb:1  # 申请1个GPU

python workflow_by_code.py --universe csi300 --only_backtest > ./logs/work—log.log 2>&1