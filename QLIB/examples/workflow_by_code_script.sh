#!/bin/bash
#SBATCH --job-name=master_model
#SBATCH --gres=gpu:2g.20gb:2  # 申请1个GPU
#SBATCH --output=output.log    # 任务输出日志
#SBATCH --error=error.log      # 错误日志
python workflow_by_code.py --universe csi300 --only_backtest