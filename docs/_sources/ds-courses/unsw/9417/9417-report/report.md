# Project - report

Machine Learning in the Unknown

Group Name: Boston Static

Group Members:

University of New South Wales

Author Note

This is a group project report for the course COMP9417 in 21T1







# Introduction

This project aims to create a machine learning model that makes the best prediction on the test set based on class-weighted F1 scores. The machine learning pipeline applied involves pre-processing, feature engineering, removing outliers, parameter tuning and selecting the final model by comparing the weighted F1-score on the validation set.

Based on the project specification, datasets that contain no domain knowledge of the features are given. CSV files are provided for training, validation and testing set, in which the test set contains only instances with no class label. The datasets consist of 8346 training and 2782 validation instances with 128 real-valued features. No headers are provided, and the target variable is composed of 6 possible values, with all features and the target variable being numerical. The test set contains 2782 data with features values only. We were required to build a model using this dataset, which is expected to accurately predict the results for the testing set.

# exploratory data analysis

This section explains the nature of the dataset such as class distribution and feature importance by graph demonstration. Given the nature of the unknown feature name, the data category is first identified via a box plot. Then value range of each column is visually evaluated to extract the underlying meaning of that feature, which helps to determine how to apply the most appropriate method.
## *A. Data distribution and feature correlation*
Based on the first diagram in Figure 1, the distribution between different classes in the target variable of the training set is not balanced. Class label 6 contains the smallest percentage of data (11.9%) and class label 2 contains the largest (21.4%). No missing values were found in the dataset, and all data was structured i.e., numerical data (Figure 2).

The second diagram in Figure 1 illustrates the top 10 most important features determined by the Extra Trees classifier[^1]. The significance of these features is computed based on the Gini importance, which is a measure of the mean impurity over all splits that includes that feature.  It shows that column 8 has the highest relative importance and all 10 features selected are above 60% relative significance. The fourth diagram illustrates the tuning for the principal component analysis, in which the maximum variance regarding the number of features converges around 10 features.  This is similar to the default value for the parameter max\_features in the tree model where the number is determined by the square root of the total number of features, which is 11.

![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.001.png) ![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.002.png) ![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.003.png)![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.004.png) 

Figure 1: target variable distribution in training set, top 10 most important features selected by extra tree classifier,  10 features by ANOVA, PCA [^2]comparison in the number of component vs variance[^3]

![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.005.png)

Figure 2: boxplot  for original data 

# methodology

The details of methods and pipeline are described below. The pipeline involves pre-processing the data, feature selection, outlier removal, parameter tuning and voting ensemble. Dimensionality reduction is applied before outlier removal. It is because if the order is shifted, additional rows would be removed due to the outliers in the redundant feature. 10 folds stratified cross-validation is applied for analysis of the variance for different models and the validation set is used as the final testing (see result section). The models are compared before and after for each processing to achieve the maximum outcome.
## *A) Select classification models*
Figure 3 illustrates the comparison of the performance for different classification models on the standardized zero-mean data. The classification models used include Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbours, Decision Tree, Naive Bayes, Support Vector Machine, Decision Tree Regressor, Random Forest, Extra Trees Classifier, and Multi-layer Perceptron. This process filters the models based on the scores and variance. Results are shown in the table below, and the models selected for the next phase are KNN, Logistic Regression, Random Forest, and Extra Trees Classifier.

`  	`![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.006.png)    ![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.007.png)![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.008.png)

Figure 3: model comparison between unstandardised and standardised data, variance and time comparison, Boxplot for models on standardised data
## *B. Feature selection*
This section discusses some feature selection techniques regarding numerical input and categorical output. 

As seen in Figure 1, the ANOVA f-test is used to select the top 10 most important features. And the result is compared with the Extra Trees classifier selection. ANOVA stands for “analysis of variance” and is a parametric statistical hypothesis that decides if the means of two or more data samples belongs to the same distribution. Such method calculates the maximum relevance between a feature and a class label, and the correlation is measured to minimise redundancy [1].

Besides, a correlation matrix of 128 columns is plotted below. It can be seen that feature No. 107 is highly correlated with the output variable y. Hence, we assigned this as a key feature for predicting the value of y. Since only one feature is found to have an absolute correlation value greater than 0.5 with y, no further step is taken to filter features during this process.

`  	`![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.009.png) ![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.010.png)![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.011.png)

Figure 4: Correlation Matrix, feature importance by Extra Trees classifier

In this process, tree-based feature selection was used instead of L1-based and PCA selection, as Lasso Regression does not evaluate whether the correct form of the relationship has been chosen between the independent and dependent variables. Furthermore, PCA is not chosen as the measurements from all the original variables are used in the projection to the lower-dimensional space, only linear relationships are considered, and PCA or SVD-based methods, as well as univariate screening methods, do not consider the potential multivariate nature of the data structure.

As seen in Figure 5, zero mean standardised data gives more accurate model performance than min-max normalisation. Based on Figure 6, PCA dimensionality reduction performs better than tree and linear SVM model selection. In general, the optimal number of the component in PCA is around 20. The scoring of the best model - Extra Trees classifier is improved to 0.9965 from 0.9955 (before the changes). (PCA is not applied in the following section due to time constraint)

`     `![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.012.png) ![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.013.png)![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.014.png)![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.015.png)

Figure 5: 10 folds cross-validation results on dimensionality reduction data by Extra Tress classifier, linear SVM (data is normalized min-max and min-max)

`  `![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.016.png)![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.017.png)

Figure 6: 10 folds cross-validation results on dimensionality reduction data by PCA, result from component = 20
## *C. Outlier removal*
As mentioned above, only numerical data is concerned in this dataset. Pre-processing for numerical features applies differently to the tree and non-tree models. In general, tree-based models do not depend on scaling for which a non-tree classifier could rely heavily on [2]. Outliers are often considered as a negative factor that skews the tendency of the data which results in poor model scoring. As a result, the outlier removal technique could lead to a more accurate model. Removing outliers involves deletion or replacement by the median value of the data points. Minimum Covariance Determinant is not considered as the data does not follow the pattern of gaussian distribution. Local Outlier Factors is not chosen due to the high dimensionality of the data. Three approaches are evaluated in this section and the results are summarised in the table below.

|Outlier method|Number of rows removed|Percentage of rows removed|
| :- | :- | :- |
|Interquartile range|2949|0.3533|
|Isolation forest|1029|0.1233|
|<h3>Density-based spatial clustering</h3>|1316|0.1577|
Figure 7: comparison between different outlier removal methods
### *1) Interquartile range*
The interquartile range is a statistical-based approach that could be applied when the data cannot be normalized by Gaussian distribution [3]. It describes the middle 50% of the data, ordering the data from smallest to largest. The first (25%) and third (75%) quartile is defined by the median value of the lower and upper half of the data. According to [4], such method often assumes that the high probability regions of a stochastic model contain the normal data, whereas the outliers stay in the low probability domain. The threshold for identifying an outlier is set to be 1.5 times the data range away from the middle quartile. As seen below box plot, the y-scale changed from 700000 to 140000 which is 5 times smaller than the original scale.

![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.018.png) ![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.019.png)

Figure 8: box plot of data after interquartile range, after isolation forest
### *2) Isolation forest*
According to [5], anomaly detection could be achieved by an unsupervised learning algorithm named isolation forest. While most model-based methods detect outliers via separating those that do not conform to the profile of normal instances, this approach explicitly isolates anomalies and is considered as a fast and low memory demand method. Based on the third graph from Figure 8, the y scale of the data reduced from 700000 to 250000 (around 3 times smaller). Compared to the interquartile range, this approach is less aggressive and preserves more pattern of the original data. 
### *3) Density-based Spatial Clustering of Application with Noise*
Another potential method for outlier removal would be Density-based Spatial Clustering of Application with Noise or DBSCAN. DBSCAN is an unsupervised, density-based algorithm, which takes features within the data set, and clusters them based on parameters. Any data points that are not within these clusters and are not ‘*density reachable*’ from any other point would be considered outliers and can be removed from the data set. By selecting features that have the largest impact on the class feature, we would then be able to find outliers within the data set concerning these selected features. Based on the graph, the y scale of the data gets reduced from 700000 to 300000, acting less aggressive than both Interquartile Range and Isolation Forest, preserving more of the original data

![](Aspose.Words.b67f9915-5a2e-4272-8f87-a899219b12ca.020.png)

Figure 9：box plot for data filtered by DBSCAN
## *D. Parameter tuning* 
10 folds cross-validation was applied for parameter tuning as it provides an unbiased estimation for the final result. Grid search and random search are two commonly used parameter tuning techniques. According to [6], a Grid search evaluates all combination of hyperparameter values and is considered as an exhaustive search among the parameters. This approach gives high accuracy result but is often time inefficient. The random search examines random combinations of the hyperparameters, which allows the users to reduce the search space. Grid search is applied to the candidate models includes the Extra Trees classifier, Random Forest, KNN, Logistic Regression. Based on the below table, the optimal nearest neighbour for KNN is 1, which is led by the high dimensionality of the data.

|Classifier|Hyperparameter s|Weighted F1 scores by 10 folds cross-validation on the training set|time|
| :-: | :-: | :-: | :-: |
|KNN|n\_neighbors=1|0.9943|10s|
|Logistic regression|C=1, max\_iter=200|0.9915|10 minutes+|
|Extra  Trees classifier|max\_features=11, min\_samples\_leaf=1, min\_samples\_split=2, n\_estimators=200, random\_state=23|0.9956|1 hour +|
|Random forest|max\_features=11, min\_samples\_leaf=1, min\_samples\_split=2, n\_estimators=200, random\_state=23|0.9951|1 hour +|
|MLP|activation='relu', hidden\_layer\_sizes=(50, 100, 50), learning\_rate='adaptive', alpha=0.0001, solver='adam'|0.9932||
Figure 10: comparison in score after parameter tuning
## *E. Voting ensemble*
To improve the performance of a single model and lower the variance, the machine learning model voting ensemble that combines the predictions from multiple other models is often considered.

A soft voting ensemble that predicts the class label based on the probability of all models, is then applied for the classification problem. The hard voting ensemble was not considered as it anticipates the result based on the most popular vote among the models, which ignores the strength and weakness of each model.

|Classifier|10 folds cross-validation on the training set|Predication on the validation set|
| :- | :- | :- |
|ExtraTrees classifier + random forest + MLP|0.9957|0.9931691|
|ExtraTrees classifier + random forest + MLP + KNN|0.9952 (variance = 0.001517)|<p>0.9931707</p><p></p>|
Figure 11: result from the voting ensemble 
## *F. Final model* 
Based on the final performance on the validation set, the voting ensemble (Extra Trees classifier + Random Forest + MLP + KNN) was chosen for the prediction for its low variance (the lowest among all candidate models,  Figure 3) and high F1-score. The hyperparameters of each model are listed above in the parameter tuning section with the unlisted ones as default. The result only differs by 3 compared with the one gained from the extra tree classifier. 

# result

The weighted F1-score is then used as the evaluation metric, and the final testing is performed on the validation set (as no testing set was provided), which is unseen to the model. All models except for MLP and tree model are trained based on standardised data via z-score. 

Based on *Figure 12*, the voting ensemble can be considered the most suitable model to apply for this dataset as it gives a balanced trade-off between the variance (overfitting issue is minimised) and the F1-score. The problem of the curse of dimensionality and outliers is minimised as the voting ensemble includes two tree models, so it is considered flexible and reliable to apply for this numerical dataset.

|Tuned Classifier |Score on the validation set (4 D.C.)|Variance of 10 folds cross-validation on the training set|
| - | - | - |
|Extra Tree classifier |0.9932|0.001721|
|Random Forest|0.9910|0.002135|
|KNN|0.9928|0.004484|
|Logistic Regression|0.9871|0.001836|
|Multi-layer Percetron|0.9932|0.002563|
|Voting Ensemble (Extra Trees classifier + Random Forest + MLP)|0.9932|0.001517|
Figure 12: Model comparison in the validation set

# discussion

It was observed that the optimal parameters for the extra tree classifier and the random forest are the same (Figure *10*), and both have a similar F1-score. This is predictable as both ensemble methods are composed of many subtrees, where the random forest applies bootstrap and the extra tree uses the whole sample space. Besides, during the runtime, the random forest chooses the optimal split whereas the extra tree does it randomly, both choosing the best subset of features after the final split point. Thus, it allows the extra tree to run faster than the random forest [7]. Given the high dimension of the data, it is understandable to observe that the tree and MLP model performs better than logistic regression and KNN. And simple model such as Naïve Bayes and linear discriminative analysis performs poorly for this dataset.

The feature importance selected by ANVOA and tree classifier is both similar. The reason for this could be the nature of feature selection primarily focus on the variance and correlation of the data. In addition, it was observed that dimensionality reduction could substantially reduce the training time for the model. However, during the testing, both outlier removal and feature selection did not improve the F1-score for the model selected. It was concluded that the chosen tree model, KNN and multi-layer perceptron deploy sufficient mechanism within its own implementation that pre-processing of the data have minimal impact on them compared to the parameter tuning effect.

# conclusion

In conclusion, this report discusses various approaches in the standard machine learning pipeline for classification problems concerning numerical data. This pipeline includes data pre-processing, such as visualising the nature of the data, feature selection, outlier removal using various methods such as Isolation Forest and Interquartile Range, parameter tuning, and comparison between the tree and non-tree models. Based on the final results gained from the validation set, the voting ensemble was determined to be the best of use in the final prediction because of its low variance and high F1 score. Some future improvements include comparing more pre-processing approaches and exploring different open-source libraries that are summarised in [8] or in many great review papers and journal articles. Additionally, more research could have been made into better methods for feature selection and outlier removal, to help tune the model to better predict test data.

# reference

The idea of pipeline and code for comparing model baseline score comes from [2]. 



|[1] |D. F. G. Zena M. Hira, "A Review of Feature Selection and Feature Extraction Methods Applied on Microarray Data," *Adv Bioinformatics,* 2015. |
| :- | :- |
|[2] |"A Complete ML Pipeline Tutorial," 2018. [Online]. Available: https://www.kaggle.com/pouryaayria/a-complete-ml-pipeline-tutorial-acu-86/notebook.|
|[3] |J. Brownlee, "How to Remove Outliers for Machine Learning," 2018. [Online]. Available: https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/.|
|[4] |I. F. Ilyas and X. Chu, in *Data Cleaning*, New York, 2019, p. 12.|
|[5] |F. T. Liu, K. M. Ting and Z. Zhou, "Isolation Forest," *2008 Eighth IEEE International Conference on Data Mining,* pp. 413 - 422, 2008. |
|[6] |P. Worcester, "A Comparison of Grid Search and Randomized Search Using Scikit Learn," Jun 2019. [Online]. Available: https://blog.usejournal.com/a-comparison-of-grid-search-and-randomized-search-using-scikit-learn-29823179bc85.|
|[7] |G. Pierre, D. Ernst and L. Wehenkel, "Extremely randomized trees," pp. 3-42, 2006. |
|[8] |J. Misiti, "Machine learning frameworks, libraries and software," 2016. [Online]. Available: https://github.com/josephmisiti/awesome-machine-learning.|
|[9] |A. Andrade and L. Golab, "DATA SCIENCE GUIDE," 2016. [Online]. Available: https://datascienceguide.github.io/exploratory-data-analysis.|
|[10] |M. Alam, "DBSCAN — a density-based unsupervised algorithm for fraud detection," 2020. [Online]. Available: https://towardsdatascience.com/dbscan-a-density-based-unsupervised-algorithm-for-fraud-detection-887c0f1016e9.|



[^1]: Extra tree stands for extremely randomized trees
[^2]: PCA stands for principal component analysis, a dimensionality-reduction method
[^3]: How far the data spreads from its mean