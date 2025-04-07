from datetime import datetime
import os
from pathlib import Path
import pickle
import sys
import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from qlib.contrib.model.pytorch_master_ts import MASTERModel
from qlib.contrib.data.dataset import MASTERTSDatasetH
from qlib.contrib.data.handler import Alpha158

from qlib.tests.data import GetData
import numpy as np

provider_uri = "~/QuantProject/.qlib/qlib_data/cn_data"
GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
qlib.init(provider_uri=provider_uri, region=REG_CN)

# 配置参数
market = "csi300"
benchmark = "SH000300"

# 数据处理器配置
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
    "infer_processors": [
        {
            "class": "RobustZScoreNorm",
            "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True
            }
        },
        {
            "class": "Fillna",
            "kwargs": {
                "fields_group": "feature"
            }
        }
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
        {
            "class": "CSRankNorm",
            "kwargs": {
                "fields_group": "label"
            }
        }
    ],
    "label": ["Ref($close, -5) / Ref($close, -1) - 1"]
}

market_data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
    "infer_processors": [
        {
            "class": "RobustZScoreNorm",
            "kwargs": {
                "fields_group": "feature",
                "clip_outlier": True
            }
        },
        {
            "class": "Fillna",
            "kwargs": {
                "fields_group": "feature"
            }
        }
    ]
}

# 模型配置
model_config = {
    "class": "MASTERModel",
    "module_path": "qlib.contrib.model.pytorch_master_ts",
    "kwargs": {
        "seed": 0,
        "n_epochs": 40,
        "lr": 0.000008,
        "train_stop_loss_thred": 0.95,
        "market": market,
        "benchmark": benchmark,
        "save_prefix": market
    }
}

# 数据集配置
dataset_config = {
    "class": "MASTERTSDatasetH",
    "module_path": "qlib.contrib.data.dataset",
    "kwargs": {
        "handler": {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": data_handler_config
        },
        "segments": {
            "train": ["2008-01-01", "2014-12-31"],
            "valid": ["2015-01-01", "2016-12-31"],
            "test": ["2017-01-01", "2020-08-01"]
        },
        "step_len": 8,
        "market_data_handler_config": market_data_handler_config
    }
}

# 投资组合分析配置
port_analysis_config = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": 30,
            "n_drop": 30
        }
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "deal_price": "close"
        }
    }
}

model = init_instance_by_config(model_config)
dataset = init_instance_by_config(dataset_config)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_name = f"{market}-{model_config["class"]}-model"
model_file_path = Path(f"./model/{model_name}.pkl")

if not os.path.exists('./model'):
    os.makedirs('./model')
    
with R.start(experiment_name="train_model") as exp:
    if not model_file_path.exists():
        try:
            R.log_params(**model_config["kwargs"])
            # 方法1：使用 sys.stdout.write
            def custom_print(*args, **kwargs):
                msg = ' '.join(map(str, args)) + '\n'
                import sys
                sys.stdout.write(msg)
            
            # 临时替换 print
            import builtins
            orig_print = builtins.print
            builtins.print = custom_print
        
            model.fit(dataset)  # 训练模型
            
        finally:
            # pkl_path = os.path.join("./model", f"{model_name}.pkl")
            # with open(pkl_path, "wb") as f:
            #     pickle.dump(model, f)
            builtins.print = orig_print  # 确保恢复原始 print
            R.save_objects(trained_model=model)
        
    else:
        model.load_model(f"./model/{model_name}.pkl")
        R.save_objects(trained_model=model)
        
    recorder = exp.get_recorder()
    rid = R.get_recorder().id
    
    all_metrics = {
        k: []
        for k in [
            "IC",
            "ICIR",
            "Rank IC",
            "Rank ICIR",
            "1day.excess_return_without_cost.annualized_return",
            "1day.excess_return_without_cost.information_ratio",
        ]
    }


    print(f"[Status]: Model Training/ Loading finished".upper())
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
        model = recorder.load_object("trained_model")

        # prediction
        recorder = R.get_recorder()
        ba_rid = recorder.id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        metrics = recorder.list_metrics()
        print(f"Metrics: {metrics}")
        for k in all_metrics.keys():
            all_metrics[k].append(metrics[k])
        print(f"All metrics: {all_metrics}")
        print(f"Available metrics: {metrics.keys()}")
        
    for k in all_metrics.keys():
            print(f"{k}: {np.mean(all_metrics[k])} +- {np.std(all_metrics[k])}")