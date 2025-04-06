# 检查MASTER环境是否已存在
if conda info --envs | grep -q "MASTER"; then
    echo "MASTER环境已存在，跳过创建步骤..."
else
    # 创建新的conda环境
    echo "创建新的MASTER环境..."
    conda create -n MASTER python=3.12
fi

# 激活MASTER环境
eval "$(conda shell.bash hook)"
conda activate MASTER

# 检查当前环境是否为MASTER
if [[ $CONDA_DEFAULT_ENV != "MASTER" ]]; then
    echo "当前环境不是MASTER，正在激活MASTER环境..."
    conda activate MASTER
fi

# 检查pip路径并设置正确的PATH
HOME_PATH=$(echo ~)
EXPECTED_PIP_PATH="$HOME_PATH/.conda/envs/MASTER/bin/pip"
CURRENT_PIP_PATH=$(which pip)

if [[ $CURRENT_PIP_PATH != $EXPECTED_PIP_PATH ]]; then
    echo "设置正确的PATH环境变量..."
    export PATH="$HOME_PATH/.conda/envs/MASTER/bin:$PATH"
fi

# install `qlib`
pip install numpy
pip install --upgrade cython

# 无论目录是否存在，都执行pip install
cd ~/QuantProject/QLIB
echo "安装qlib开发版本..."
pip install -e .[dev]
cd -

python -m qlib.install init
pip install -r requirements.txt


#

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
if [ ! -d "./backtest" ]; then
    mkdir ./backtest
fi

sbatch job_submit.sh
squeue -u 24039378g