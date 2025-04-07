
# Quantitative Finance Project: Investment Trend Forecasting Based on Qlib

This repository contains a quantitative finance project that utilizes the Qlib library to predict stock return trends, analyze quantitative strategies, and compare the performance of three models: LightGBM, XGBoost, and MASTER. The project focuses on developing a robust quantitative investment strategy on the CSI 300 index.

---

## Features
- **Data Handling:** Utilizes Qlib's built-in datasets, such as Alpha158, for data preprocessing, alignment, and factor engineering.
- **Models Used:**
  - **LightGBM:** Efficient gradient boosting framework for tabular data.
  - **XGBoost:** Scalable tree boosting library for regression tasks.
  - **MASTER:** Deep learning-based transformer model for sequential stock prediction.
- **Visualization:** Provides insightful charts, including cumulative returns, risk metrics, and IC reports, to evaluate the strategy's performance.
- **Backtesting:** Conducts realistic backtesting of investment strategies, highlighting transaction costs, risk adjustment, and excess returns.

---

## Key Results
- **Strong cumulative returns:** The long-short strategy consistently outperforms benchmarks with high excess returns.
- **Robust IC performance:** The IC values show near-normal distribution and high predictive validity.
- **High autocorrelation:** Indicates continuity and predictability in strategy returns.
- **Challenges:** IC values exhibit seasonal patterns and a weakening trend in 2020, signaling potential strategy attenuation.

---

## File Structure
### Main Workflow
The implementation is located in:  
`QuantProject/QLIB/examples/workflow_collection`

### Key Directories
- `data`: Contains processed datasets in binary format.
- `models`: Stores configurations and evaluation results for the MASTER, LightGBM, and XGBoost models.
- `visualizations`: Includes plots of cumulative returns, risk metrics, IC distributions, and autocorrelations for comprehensive analysis.

### Report
The full analysis is detailed in `Quantitative Finance Project Report_ Investment Trend Forecasting Based on Qlib.pdf`.

---

## Installation and Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Qlib**:
   Follow instructions from the [Qlib documentation](https://qlib.readthedocs.io/en/latest/).

4. **Run the workflow**:
   Navigate to the workflow directory:
   ```bash
   cd QuantProject/QLIB/examples/workflow_collection
   ```

---

## References
1. [LightGBM](https://github.com/microsoft/LightGBM)
2. [XGBoost](https://arxiv.org/abs/1603.02754)
3. [MASTER Paper](https://arxiv.org/abs/2312.15235)

For detailed insights, please refer to the `Report` PDF included in the repository.

---

## Contact
For issues or inquiries, please open a GitHub issue or contact the project maintainers.
