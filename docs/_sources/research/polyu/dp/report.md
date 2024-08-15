# Data Pruning

Github Repo Link: https://github.com/postsent/dataPruning4Rec

## Method

<br />
<br />

![](overview.jpg)

<center>Three stages in data pruning: model training, scoring and post sampling.</center>

<br />
<br />
<br />

![](recsample_pipeline.jpg)

<center>Our pipeline in sampling divese high quality recommendation data.</center>



![](paper.jpg)

## Experiment

## Main

MoE + Pruning gives the best performance with minimal training requirement.

<center>Recall@20/100.</center>

![](main.jpg)

<center>MRR@20/100.</center>

![](main_mrr.jpg)

## MoE Performance

MoE perform well in both Recall (if recommended item is in topk) and MRR metrics (how close the rank is to top1)

![](moe_perf.jpg)

### Ablation

<center>MoE vs Ensemble vs Single model on data pruning.</center>
Keep 30% data for Yoochoose, 60% for Amazon-M2. AMZ-M2 is much larger thus requires more data to have close performance as if it is trained on the full dataset.
<br />
<br />

![](abla1.jpg)

![](abla2.jpg)

### Dataset Statistics

![](data.jpg)

![](more_data.jpg)

### EDA

![](metrics_dff_data.jpg)

We use AUM as the heuristic to filter data. Different data quality metrics perform differently depending on the dataset applied to. In general, AUM gives on avg the best results since the data subset have a health coverage on both easy (left) and hard (right) and medium samples (middle) i.e. the bell shape. 

![](pred_overlap.jpg)

Model prediction can complement each other since each has its own strength. Prediction overlap for different models 0~4: transformer, attention based, GNN, GRU, self attention. 
