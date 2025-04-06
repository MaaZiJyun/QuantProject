from qlib.data.filter import BaseDFilter
from datetime import datetime, timedelta

class ListAgeFilter(BaseDFilter):
    def __init__(self, min_days=365, fstart_time=None, fend_time=None):
        self.min_days = min_days
        super().__init__(fstart_time, fend_time)
    
    def filter_main(self, instruments, start_time=None, end_time=None):
        filtered = {}
        for inst, data in instruments.items():
            list_date = datetime.strptime(data["list_date"], "%Y-%m-%d")  # 假设有上市日期字段
            trade_date = datetime.strptime(end_time, "%Y-%m-%d")  # 使用筛选结束时间作为当前日期
            if (trade_date - list_date).days >= self.min_days:
                filtered[inst] = data
        return filtered