# Report – Text Analysis with Spark RDD API
for COMP5349 Assignment 1 
2022.04.14

# Design thought process and read data
The overall design focuses on reusing intermediate resilient distributed datasets (RDD) results as well as breaking large data into atomic units for parallelism and load balancing purposes. Note that only the summation and subtraction process is suitable for breaking down and adding back up. Therefore, summation related processes are broken down as much as possible and then a final division is applied at the very end. Document (clause) and candidate phrase are treated as atomic units in method 1 while candidate phrase and word are for method 2. All data transformation is processed with RDD while detailed data manipulation is achieved with the user supply function in Python. 

The dataset involved two CSV files governing law and anti-assignment, which are read via the Spark CSV API. After reading, two RDDs are created for each file. A filter transformation is then applied to both RDDs to extract all the rows from the three columns. This process creates three new RDDs, which are required for later computation. To remove the row with the nan or no value, a filter transformation is applied to each aforementioned RDDs. These three RDDs are then used as the input for the two methods introduced in the paper named algorithm automatic keyword extraction from individual documents [1]. Three RDDs are treated separately to obtain the final result in both method 1 and 2.

The Spark transformation [2] used for both methods are map, reduceByKey, mapValues, distinct, sortBy, groupByKey, join, filter and flatMap. Spark Action used is take.
# Methods 1
First, a map and reduceByKey transformation are used for the input RDD to make the row element inside the RDD iterable. As seen below, every row element in the input RDD will be put into a list which will be stored as a value inside a new RDD where the key is 1.

row\_num\_pre = no\_nan.map(lambda row: (1, [row])).reduceByKey(lambda a, b: a + b) 

Second, a flatMap transformation is applied to the new RDD to loop over the list of row elements and assign a unique id to each. This forms another RDD that has a key of unique id with a value of the row element. 

row\_num = row\_num\_pre.flatMap(lambda x: assign\_row\_num(x, col))

Third, another flatMap transformation is applied to the second RDD to split the row element into different documents based on the page number. It leads to a new RDD that contains an element with the key of an id and the value of a list of tokenized documents (list of words).

clause\_list = row\_num.flatMap(split\_clause) 

Fourth, a map transformation is applied to the previous RDD to obtain the candidate phrases which is separated by stop words or punctions. It creates a new RDD composed of elements with the key of an id with values of candidate phrases, the content words and word frequency for the given document. This RDD prepares all the data required for later calculation.

cand\_phrases = clause\_list.map(get\_candidate\_phrase)

Fifth, we use a flatMap transformation to the aforementioned RDD to calculate the score for each word and then the candidate phrase. This returns a new RDD made of elements with key as id, value as candidate phrase and score.

cand\_phrase\_score = cand\_phrases.flatMap(calculate\_score\_1)

Sixth, the fifth RDD is mapped to a new RDD with key as id and value of a tuple consisting of candidate phrases and score.

cand\_phrase\_score\_remap = cand\_phrase\_score.map( lambda x: (x[0], (x[1], x[2]) ) )

Seventh, the previous RDD is grouped based on the id with a value of the aforementioned tuple, which creates a new RDD.

cand\_phrase\_score\_groupkey = cand\_phrase\_score\_remap.groupByKey().mapValues(list)

Eighth, the returned RDD is then transformed by the flatMap into a new RDD. It involves extracting the top 4 candidate phrases based on the corresponding score. This RDD’s elements have a key of phrase and value of id.

cand\_phrases\_top4=cand\_phrase\_score\_groupkey.flatMap(extract\_cand\_phrase\_top4)

Ninth, a new RDD with a key as a phrase and value of id is formed by remapping the fifth RDD. 

cand\_phrases\_duplicate = cand\_phrase\_score.map(lambda x: (' '.join(x[1]), x[0]))

Tenth, groupByKey and map transformation is applied to obtain the relative document frequency (rdf).

rdf = cand\_phrases\_duplicate.groupByKey().map(calculate\_rdf)

Eleventh, we create a new RDD with a key of phrase and value of -1 which is a dummy from the ninth RDD. Distinct and map transformation is applied.

cand\_phrases\_unique = cand\_phrases\_duplicate.keys().distinct().map(lambda x: (x, -1))

Twelfth, we use the join, groupByKey and mapValues transformation on the eleventh RDD to obtain only the top 4 keywords for each document. The new RDD has a key of phrase and value of tuple with id and -1.

edf\_join= cand\_phrases\_top4.join(cand\_phrases\_unique).groupByKey().mapValues(list)

Thirteenth, the extracted document frequency (edf) is then the length of the list that is inside the twelfth RDD. The new RDD formed has the key of the phrase and value of tuple with the edf and string edf for late identification.

edf = edf\_join.map(lambda x: (x[0], (len(x[1]), 'edf')))

Fourteenth, the Thirteenth RDD is joined to the Tenth RDD so as to prepare data for calculating the essentiality (ess) of a keyword.

ess\_pre = edf.join(rdf)

Fifteenth, the essentiality is calculated for each keyword and the new RDD has a key of a phrase and value of rdf, edf, and ess.

ess = ess\_pre.map(calculate\_ess)

Sixteenth, the final output is obtained by sortBy transformation and the take action which selects the top 20 keywords based on the ess value.

res = ess.sortBy(lambda r: r[3],ascending=False).take(20)

More intuitive sequential and parallel relationships between all transformations are illustrated in **Table 1**.

# Methods 2

First, a map transformation is used for the input RDD to pre-process the input RDD which includes tokenisation, page number removal and conversion to lowercase. It creates a new RDD that has a value of a list of lists of tokens.

clean = clean.map(lambda x: preprocess(x, col))

Second, a flatMap transformation is applied to the first RDD to extract the candidate phrase. This forms another RDD that has a key of the actual phrase with a value of -1. 

cand\_phrases\_duplicate = clean.flatMap(get\_cand\_phrases\_list)

Third, another flatMap and reduceByKey transformation is utilised in the second RDD to get the frequency of each word that appears in the given candidate phrase. It leads to a new RDD that has a key of a word and a value of its frequency.

freq = clean.flatMap(get\_freq).reduceByKey(lambda a, b: a + b) 

Fourth, a flatMap and reduceByKey transformation is used to the second RDD to obtain the candidate phrases. It creates a new RDD composed of elements with the key of a word and a value of its co-occurrence.

cand\_phrases\_co\_occurance = cand\_phrases\_duplicate.flatMap(get\_co\_occurence).reduceByKey(lambda a, b: a + b)

Fifth, we use a join transformation to the third and fourth RDD to prepare the dataset for calculating the score for each word. This returns a new RDD with key as a word and value as a tuple with frequency and co-occurrence.

score\_join = freq.join(cand\_phrases\_co\_occurance)

Sixth, the fifth RDD is mapped to a new RDD to calculate the score for each content word. It makes a new RDD with the key as a word and the value as the score.

goven\_score\_res = score\_join.map(calculate\_score\_2)

Seventh, the second RDD is grouped based on the phrase and distinct transformation is also applied to form a new RDD that has a key of the unique phrase and value of -1.

cand\_phrases\_unique = cand\_phrases\_duplicate.groupByKey().keys().distinct().map(lambda x: (x, -1))

Eighth, the previous RDD is then transformed by the map into a new RDD. It simply converts the list of tokens to a string. This RDD’s elements have a value of a phrase.

cand\_phrases\_unique = cand\_phrases\_unique.map(lambda x: x[0].split(' '))

Ninth, the eighth RDD is transformed with a map and reduceByKey to a new RDD. It has a key of 1 with a value of a list of phrases. 

goven\_phrase\_iterable = cand\_phrases\_unique.map(lambda x: (1, [x])).reduceByKey(lambda a, b: a + b) 

Tenth, flatMap transformation is applied to assign a unique id to each phrase. It creates an RDD with a key of a word with a value of a phrase.

goven\_phrase\_unpacked = goven\_phrase\_iterable.flatMap(assign\_phrase\_id)

Eleventh, we join the tenth and sixth RDD followed by a map transformation to form a new RDD with a key of a phrase and a value of a tuple made of score and word.

goven\_phrase\_join\_score = goven\_phrase\_unpacked.join(goven\_score\_res).map(lambda x: ( x[1][0], (x[1][1], x[0]) ) )

Twelfth, we use groupByKey and mapValues transformation to obtain the score for each phrase. Note we ignore duplicate words in the phrase and only the word score once. It creates a new RDD with key as phrase and value as a list of tuples composed of score and word.

phrase\_score = goven\_phrase\_join\_score.groupByKey().mapValues(list).map(get\_unique\_score\_4\_phrase)

Lastly, the final output is obtained by sortBy transformation and the take action which selects the top 20 keywords based on the score.

phrase\_score.sortBy(lambda r: r[1],ascending=False).take(20)

More intuitive sequential and parallel relationships between all transformations are illustrated in **Table 2**.
























# Table section
## Table 1
Note that each clause is treated as one document in table 1.

Note that for the below table, -1 is used for the dummy because join transformation requires a key-value pair. The table is used for a concise demonstration of the data flow. The -> symbol is used when a step involves more than one transforming a result. 



|Step|Transformation|Purpose|Input|<p>Output RDD Format</p><p></p>|Explanation|
| :- | :- | :- | :- | :- | :- |
|s1|filter|Preproce|csv read|row(...)|Remove nan and none value in a row|
|s2|map, reduceByKey|Make row iterable|s1|(1, [row1, …])|Put the row into a list which is a value for a key-value pair where the key is 1 (dummy).  In this way, we can loop through the row via map and assign it a unique id.|
|s3|flatMap|Split clause|s2|(clauseId:str, clause:list)|First, preprocess the data (tokenization, lowercase). Then, different phrases are extracted based on the page number (this value is then discarded). Next, a unique id is assigned to each clause for later identification. The id is in the form: "fileNumber-NthClause" (in that row).|
|s4|map|Candidate phrase|s3|(clauseId, phrase:str, content word:list, frequency:dict)|For each clause. extract the candidate phrase, content words and word frequency for later computation convenience.|
|s5|flatMap|Score|s4|(clauseId, phrase, score) -> (clauseId, (phrase, score))|Compute the score and create another RDD to store the key value pair form.|
|s6|<p>groupByKey,mapValues</p><p>(list)</p>|Top 4 keywords for each document|s5|<p>` `(clauseId, </p><p>[(phrase1, score),..])</p>|Group by clause id, then select the top 4 candidate phrases as keywords based on the score. No duplicate is allowed.|
|s7|map|Collect all candidate phrase|s5|(phrase:str, clauseId)|Collect all candidate phrases, duplicate key allowed.|
|s8|groupByKey, map|rdf|s7|<p>(phrase:str, </p><p>(rdf:int, "rdf" ))</p>|<p>Group all phrases and then count the unique clause id., which is then the value of rdf for that phrase.</p><p></p><p></p>|
|s9|map, distinct|Unique phrase|s5|(phrase, -1)|Collect all unique candidate phrases.|
|s10|join|edf|s6, s9|<p>(phrase:str, </p><p>(edf:int, "edf" ))</p>|After the join, only the candidate phrase that exists in the top 4 of documents will be retained. A count is then applied for calculating edf. Note both s6 and s9 have unique keys.|
|s11|join|ess|s8, s10|<p>(phrase:str, </p><p>(</p><p>(edf, "edf"),</p><p>(rdf, "rdf" )</p><p>) -></p><p>(phrase, rdf, edf, ess)</p>|Join edf with edf based on the phrase and obtain the needed data for calculating the ess value and then calculate it.|
|s12|sortBy|Top 20 keywords for each corpus||(phrase, rdf, edf, ess)|select the top 20 keywords for each corpus based on the ess score from highest to lowest.|
Table 1: Transformation and operation involved in method 1 with a concise explanation















## Table 2

|Step|Transformation|Purpose|<p>Output RDD Format</p><p></p>|Explanation|
| :- | :- | :- | :- | :- |
|s1|filter|Preprocess|row(...)|Remove nan and none value in a row|
|s2|map|Preprocess |phrase:list|Preprocess the data (tokenization, lowercase, lemmatization )|
|s3|flatMap|Candidate Phrase (with duplicate)|(phrase:str, -1)|Get candidate phrases, and add a dummy -1 to make it a key-value pair for later operation. Duplicate phrase since it is used for score calculation. The candidate phrase is converted to a string from the list of tokens which then can be used as a key to join other RDD.|
|s4|flatMap, reduceByKey|Frequency |(word:str, frequency:float)|Calculate the frequency of each word in a given row and flatten the result. This result is then reduced based on the key to sum up the total frequency for each word.|
|s5|flatMap|co-occurrence|(word:str, co-occurrence:float)|For each element in the candidate phrase RDD (s3), we calculate the co-occurrence for each word in the given element.|
|s6|join|Degree|phrase:str, <br>(frequency:float, co-occurrence:float)|Join the frequency RDD with the co-occurrence based on the candidate phrase key.|
|s7|map|Score|(word:str, score:float)|Calculate the score for each word given the frequency and co-occurrence.|
|s8|groupByKey, distinct, map|Unique candidate phrase|(phrase:str, -1) -> phrase:list|<p>Collect all unique candidate phrases based on s3. Then convert them to a list of tokens.</p><p></p><p></p>|
|s9|map, reduceByKey|Make phrase iterable|(1, [phrase1:list, …])|Make the candidate phrase as a list in a value so it is iterable. Then we can loop through the phrase via map and assign it a unique id.|
|s10|flatMap|Assign unique id to a word|(word:str, phrase:str)|For every phrase in s10, for every word in a phrase, we extract the word out and assign it a unique id which is the phrase it belongs to.|
|s11|join|scores for phrase|(word, (phrase, score)) -> (phrase, score)|Join the s11 with s7 and obtain the score for each phrase that the word belongs to.|
|s12|groupByKey,mapValues(list), map|Unique score for each phrase|(phrase, score)|Sum up the score for each phrase based on the phrase key. Check if duplicate words are in a phrase, if so, unique is used.|
|s13|sortBy|Top 20 keywords for each corpus|(phrase, score)|select the top 20 keywords for each corpus based on the score from highest to lowest.|

Table 2: Transformation and operation involved in method 2 with a concise explanation



























# Assumption and edge case
Some of the below is based on Ed posts where a quotation mark will then be used.

1. When ordering based on the score, if the score is the same, then no action is taken, use the default behaviour of Spark API orderBy function.
1. The score of identical phrases from different documents will not be added up.
1. Deal with hyphen e.g.  "conflict-of-laws", in this case, the default behaviour of work tokenizer from nltk is applied where the hyphen is retained. Similarly, “U.S.A " will use the default API behaviour where the word is kept as it is.  "and/or" are added to the stop words.
1. Duplicate words in the same candidate phrase e.g. score of "many many people"  will equal to score of "many" + score of "people" as suggested on Ed.
1. Clauses are separated by (Page x) or (Page x-y) where x and y are page numbers.
1. “For the rdf, edf and ess calculation, we do not consider candidate phrases that have been rejected due to their lengths” based on Ed.
1. “We do not consider sub-words. For example, "agreement" and "agreements" are deemed to be totally different words.”.


# Reference
[1] Rose, S., Engel, D., Cramer, N. and Cowley, W., 2010. Automatic keyword extraction from individual documents. Text mining: applications and theory, 1, pp.1-20.

[2] "RDD Programming Guide - Spark 3.2.1 Documentation," *spark.apache.org*. https://spark.apache.org/docs/latest/rdd-programming-guide.html#basics (accessed Apr. 14, 2022).


