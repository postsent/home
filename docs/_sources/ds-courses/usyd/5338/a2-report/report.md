COMP5338 Assignment 2		Jerry Wei
# **Performance Observation Task**

# **Introduction** 

This report aims to evaluate the performance impact of different query designs and types of indexing such as the single field, compound and multi-key index in the MongoDB system. An alternative improved implementation is made for workload 5 which investigates some potential factors that could improve the query performance. 

# **Single field index and compound index usage** 

Query 1 evaluates two query plans as demonstrated in the first two images in Figure 1 which are one winning plan and one rejected plan respectively. The winning plan involves 2 stages, namely, FETCH and IXSCAN stages. The former one is to retrieve documents from the database and the latter one is to scan the index key for query optimisation. In the second stage, the key used is yearlyAvgDryDays which is a single field index that is not unique, sparse or partial. The direction of index traversal is backward due to the descending sort order in the query. The associated index bound is between 150 (exclusive) and positive infinity. Similarly, the rejected plan also involves two stages i.e. FETCH and IXSCAN. The key used is the compound index of yearlyAvgDryDays and yearlyAvgSnowdays that is not unique, sparse or partial. The index traversal direction is backward and the index bound for the key yearlyAvgDryDays is between 150 (exclusive) and positive infinity. Furthermore, the yearlyAvgSnowdays index has a full index range bound between the maximum key and minimum key since it is not directly involved in the query. 

The winning plan is chosen because the single field index helps run the query faster as a result of its smaller index size compared to the compound index. Note that the MongoDB query optimiser chooses the optimal plan by running all candidate plans partially in parallel and then selects the plan that first returns all the desired items. This faster performance by the single field index is because the compound index bound involves two fields which causes additional time required for loading the additional yearlyAvgSnowdays field when only the yearlyAvgDryDays field is actually needed.

The yearlyAvgDrydays single field index is not evaluated because the query does not contain a field that requires data from the yearlyAvgDrydays field.

The sort order can be obtained from the index for the query. Based on MongoDB's official user manual, the inclusion of a stage called SORT is an indicator that the sort order cannot be obtained from the index and need to be handled in memory. As mentioned before, the stages involved in the winning plan do not contain a SORT stage which means the sort order is obtained from the index.

Query 2, in contrast, evaluates only one plan with no rejected plan as shown in the last two images in Figure 1. There are 6 stages in total involved in the query plan which are SUBPLAN, SORT, FETCH, OR, IXSCAN and IXSCAN, respectively. Note that SUBPLAN refers to a generic index scan. After SUBPLAN stage, the query optimiser selects a sort pattern based on the query (descending order for both fields in this case) and then runs the sort operation in memory once the relevant input is received. Next, the optimiser calculates the memory limit based on the hardware specification. After that, the data is retrieved from the database and the system creates an OR stage to evaluate the two conditions from the query. Note that the OR stage will evaluate the two conditions in the order as shown in the third image in Figure 1 i.e. if the first condition is satisfied, then the second children stage will not be evaluated by the query optimiser. Next, the first children stage of the OR stage is IXSCAN with a yearlyAvgSnowdays single field index, which uses forward direction index traversal with an index bound from 150 to positive infinite. The second children stage of the OR stage is another IXSCAN with a compound index. It also uses forward direction index traversal with an index bound from 100 to positive infinite for the yearlyAvgDryDays field and minimum and maximum key for the other field. 

The single field index yearlyAvgSnowdays is not evaluated because the compound index is prefered by the query optimiser. It is because the compound index includes the yearlyAvgSnowdays as its first field meaning that there will be no performance loss when using it over the single field and the additional field gives more capability in improving the query performance and thus is chosen by the query optimiser. 

The sort order cannot be obtained from the index as shown in the third image in Figure 1 where the execution stages include a SORT stage that uses a blocking sort operation which reads all input documents. The reason is that the execution of the OR stage leads to a smaller remaining document size compared to the compound index size, which makes in-memory sorting more efficient than using the compound index. Therefore, in memory sort is prefered by the query optimiser.



In terms of execution statistics, the differences between the two queries are 1) the number of plans evaluated i.e. two compared to one. 2) the number of indexes used i.e. one compared to two 3) the number of stages taken i.e. two compared to six. The yearlyAvgDrydays index is evaluated in the first query whereas the yearlyAvgSnowdays index and the compound index are used in the second query because 1) the index can only be used when they exist in the query field 2) if multiple indexes could be chosen, then the better performance one, in terms of space and time complexity, will be selected.

`  `![](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.001.png) ![A picture containing timeline Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.002.png) ![Graphical user interface, text, application Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.003.png) ![](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.004.png)

*Figure 1 – Query 1 & query 2 winning and rejected plans*
# **Multikey Index Usage** 

There is no rejected plan but one winning plan in query 3 as shown in the first two images in Figure 2. The winning plan involves two stages called FETCH and IXSCAN. Firstly, the query optimiser defines a filter stage with a condition on the dryDays field in each array element in the monthlyAvg field by two range bounds. Another condition is set on the region field based on the value United States. Note that this filter order is different to those in the query due to the query optimisation. Next, the IXSCAN stage uses a multikey index with a forward direction traversal and a bound between 20 and 30 on the dryDays field. This stage will be used to optimise the aforementioned filter stage with the relevant index. Because the query uses the elemMatch operator to join the predicates, the MongoDB query optimiser can intersect the upper and lower bounds to produce a smaller range document scan.

There are two plans evaluated in query 4 where one is rejected and one is winning as shown in the last two images in Figure 2. The winning plan has two stages i.e. FETCH and IXSCAN. In the FETCH stage, a filter operation is defined with a condition on the dryDays field with only one bound i.e. less than 30 and another condition on the region field based on the value United States. In the IXSCAN stage, it uses a multiKey index with a forward direction traversal with a bound between 20 and positive infinity on the dryDays field.  Since query 4 does not use elemMatch operator, the multikey index bounds cannot be intersected. This means when the query is run, the system will 1) match all elements that are less than 20 without using an index 2) filter region by the United States 3) use an index to filter the remaining documents with a value higher than 20. The rejected plan involves FETCH and IXSCAN stage, which is the same as the winning plan. The FETCH stage is similar to the winning plan but it filters the documents based on the condition greater than 20 on the dryDays field instead. And that the IXSCAN stage is also similar except the bound is between 30 and negative infinity. The plan is rejected because the less than 30 bound in the winning plan gives fewer documents size which is considered more efficient.

![](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.005.png)![](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.006.png)![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.007.png)![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.008.png)

*Figure 2 - query 3 & query 4 winning and rejected plan*

# **Aggregation Performance** 

In summary, the alternative implementation has all SORT stages replaced by the index scan and an additional projection stage which reduces the initial input document size by around 25%. Given the sort order can be obtained from the indexes, the number of documents scanned and memory usage is reduced to 0 relatively for the associated sort stage.

Note that the performance comparison below shows the execution statistics in a stage-wise manner. It is because the explain method tool gives details on the first stage but less on the others. To access a more detailed explanation in the intermediate stage, the output before the investigated stage is stored as a collection and the investigated query stage is run separately based on the saved collection. Additionally, to ensure fairness in the comparison, all cached query plans are removed every time for each investigated execution. Besides, to remove the instability in query running time, standard ablation study practice is applied with 5 experiments run independently where the mean and standard deviation is recorded. In practice, however, the execution time is averaging at 2 milliseconds and thus the time cost is not further compared below. The alternative implementation is provided in Figure 5. 

In the original implementation, the execution statistics involves mainly 5 stages.  The query 1) flatten the monthlyAvg field and returns 1260 documents 2) group by city field and aggregate the low, rainfall and month field into an array, which uses 476700 bytes of memory and is executed in  2 milliseconds 3) use a projection operation to create two new fields. The lowestMinTmpMonth field will first sort the monthlyAvg array field by the low and then the rainfall field in ascending and descending order respectively. Then, it collects only the first element of the sorted array. The wettestMonth field runs similarly except the sorting is based on descending and ascending order on the rainfall and low field. This stage returns 105 documents and takes 3 milliseconds. 4) a match operation is run based on the equality between the month in the lowestMinTmpMonth and wettestMonth, which outputs 12 documents in 3 milliseconds 5) the documents are sorted based on the \_id field in ascending order in 3 milliseconds with approximately 11968 bytes data involved.

The following section highlights why the alternative implementation is more efficient.

Firstly, an improvement made is to use projection operation to reduce scanned document size as shown in Figure 3. Since only four fields, namely, city, low, rainfall and month are needed for query 5 input, all other unnecessary fields are filtered using the project. This helps reduce the storage size from 64 to 48 MB and the average object size from 159 to 100 MB.

Secondly, a single field city index is used to speed up the sorting operation as shown in Figure 4. It is beneficial because the sort order can be obtained from the index and no in-memory sort is needed. Note that the \_id field is replaced by the city value and we assumed the replaced \_id index will not be used in the sorting by default. The memory usage for that particular stage is reduced from 104857600 bytes / 1024 / 1024 = 100 MB to 0 since no in-memory sort is needed. This is demonstrated by the memLimit field in the first image of Figure 4.

Thirdly, two compound indexes are made to optimise the two-field sorting stage. The change from in-memory sort to index scan reduces the number of documents scanned size from 100 Mb to 0. Similarly, memory usage is also reduced to 0  relatively for sorting. To access detailed execution statistics and consider the sortArray operation has the same behaviours as a push followed by a sort operation ( based on the MongoDB official website),  we decompose the sortArray operation using unwind and sort operations as shown in Figure 6. Besides, the reason why two compound indexes are created in different order is because the order of the two sorting fields in the query is also different. To elaborate, the sorting order of the compound field is global relative to the whole collection on the first field but local on the second.

Fourthly, a projection operation is placed as the last stage to remove the redundant data in which the original implementation has the lowestMinTmpMonth and wettestMonth holds the same data. The final result output is illustrated in Figure 8.

Lastly, a potential improvement is to optimise the two-field sorting stage, which is not implemented. Given the two-field sorting stage needs only one element as output as suggested by the first operator, we could optimise the query by splitting the sorting into 3 stages: 1) sort by the first field as normal 2) iterate through the sorted array in the first-field sorting order and start filtering out array elements once it has a different first field value 3) sort the remaining array by the second field. In this way, we do not need to sort the whole array by the second field, but rather only those with the same first appeared first-field value. Given the small likelihood of array elements with the same first field value, this could reduce the document scanned size from 12 to potentially 1 for each document for this particular stage.

![](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.009.png)![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.010.png)

*Figure 3 – comparison with and without projection*

![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.011.png)![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.012.png)

*Figure 4 – comparison without and with city index*

![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.013.png)

*Figure 5 - an alternative implementation*

![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.014.png) ![A picture containing text

Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.015.png) 

*Figure 6 - decompose the sortArray to sort operation to explain the index effect*

![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.016.png)![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.017.png)![Text Description automatically generated](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.018.png)![](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.019.png)

*Figure 7 - before and after adding two compound index comparison*

![](Aspose.Words.227f6b1f-b9cf-4815-828b-323521354788.020.png)

*Figure 8 – final output with same data but different structure*
# **Conclusion**

In conclusion, using explicit projection, single field, compound and multi-key index under appropriate conditions could reduce the number of documents scanned and memory usage significantly, which helps improve the query performance. This is examined and demonstrated by performance comparison in the different experiments mentioned above.


2

