# Dataset: JIRA Defect Prediction Benchmark (65 Metrics)

This directory contains the file-level software defect prediction dataset evaluated in the **EGBE** study.

## Dataset Overview
- **Projects**: 9 open-source Apache software systems:
  - ActiveMQ
  - Camel
  - Derby
  - Geronimo
  - HBase
  - Jfreechart
  - Lucene
  - Mahout
  - Wicket
- **Total Releases**: 32 distinct release versions
- **Total File-Level Instances**: 73,395 modules
- **Class Imbalance**: Severe (~9.78% overall defect rate)
- **Target Feature**: `RealBug` (Binary: `1` = Defective, `0` = Clean)

## Metric Dimensions (65 Metrics)
1. **Static Code Structure (43 metrics)**:
   - Chidamber & Kemerer (CK) Suite: WMC, DIT, NOC, CBO, RFC, LCOM, LCOM3
   - Traditional Line & Complexity Counts: LOC, McCabe Cyclomatic Complexity, Max/Avg Nesting
   - Halstead Suite: Volume, Difficulty, Effort, Bugs estimate, etc.
2. **Process Churn (17 metrics)**:
   - Added lines, Deleted lines, Total Churn, Commits count, COMM, etc.
3. **Developer Ownership & Experience (5 metrics)**:
   - `OWN`: Degree of code ownership by primary author
   - `MINOR`: Count of minor contributors
   - `ADEV` / `DDEV`: Active and distinct developer counts
   - `EXP`: Developer prior experience

## Download Instructions
The standardized 32-release dataset with binary target labels and 65-metric predictor matrices is archived on Kaggle:
- **Kaggle Dataset**: [JIRA Defect Datasets Full](https://www.kaggle.com/datasets/efrathhossainshihab/jira-defect-datasets-full)

Place downloaded CSV files in this directory (`data/`):
```text
data/
├── README.md
├── activemq-5.0.0.csv
├── activemq-5.1.0.csv
├── camel-1.4.0.csv
├── derby-10.2.1.6.csv
└── ...
```

## References
- S. Yatish, J. Jiarpakdee, P. Thongtanunam, and C. Tantithamthavorn, "Mining software defects: Should we consider affected releases?" in *Proc. ICSE*, 2019, pp. 654–665.
- C. Tantithamthavorn et al., "The Large Defect Prediction Benchmark," 2022.
