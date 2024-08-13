# Data Pruning

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

<center>Recall@20/100.</center>

![](main.jpg)

<center>MRR@20/100.</center>

![](main_mrr.jpg)

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