# Characters, Spirals and Hidden Unit Dynamics 

[Question](https://www.cse.unsw.edu.au/~cs9444/21T2/hw1/)

# Contents
[Question 1 - NetLin	1](#_Toc76828508)

[Question 2 - NetFull	2](#_Toc76828509)

[Question 3 - NetConv	2](#_Toc76828510)

[Question 4 – result discussion	2](#_Toc76828511)

[**PART 2 - Rectangular Spirals Task**	4](#_Toc76828512)

[Question 1 - code	4](#_Toc76828513)

[Question 2 - code	4](#_Toc76828514)

[Question 3 – show result picture (1 hidden layer)	4](#_Toc76828515)

[Question 4 - show result picture (2 hidden layers)	5](#_Toc76828516)

[Question 5 – result discussion	6](#_Toc76828517)

[**PART 3 - Encoder Networks**	7](#_Toc76828518)

[**PART 4 - Hidden Unit Dynamics for Recurrent Networks**	9](#_Toc76828519)

[Question 1 – SRN, illustrate hidden state	9](#_Toc76828520)

[Question 2 - an bn	9](#_Toc76828521)

[Question 3 - result discussion	10](#_Toc76828522)

[Question 4 - an bn cn	11](#_Toc76828523)

[Question 5 – result discussion	11](#_Toc76828524)

[Question 6 – LSTM, analysis behaviour  and explain	12](#_Toc76828525)

[APPENDIX	15](#_Toc76828526)



**PART 1- Japanese Character Recognition**
# Question 1 - NetLin
**Confusion matrix and Final Accuracy**

![P25#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.001.png)

# Question 2 - NetFull
Process of tuning the hyperparameters see Appendix A.

**Number of hidden nodes = 200**

![P30#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.002.png)
# Question 3 - NetConv

Process of tuning the hyperparameters see Appendix A.

` `**He weight initialisation + mom=0.9 + kernel size = 3, lr = 0.01**, others as default

![P35#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.003.png)
# Question 4 – result discussion

***a) Relative accuracy analysis***

It can be seen from the above result that the order of performance from highest to lowest is:

|order|Model |Test accuracy (Epoch 10)|
| - | - | - |
|1|2 convolution layers with one fully connected layer|94%|
|2|2 fully connected layers (using tanh at the hidden nodes and log softmax at the output)|85%|
|3|1 fully connected layer (log softmax)|70%|

It can be observed that the deeper and more complex the network is, the higher the test accuracy. 

Firstly, a deeper network contains more parameters and can learn more intermediate features of the given image, which gives a better generalisation across data. 

Secondly, the character classification task here involves 70,000 images with many different writing styles which is hard for the model such as NetLin to classify. It is because it only contains one fully connected layer and so can only distinguish linearly separable features, which is not the case here.

NetFull model, on the other hand, has a non-linear activation function tanh and two fully connected layers, which uses more parameters to learn more features than NetLin. Thus, it gives higher accuracy than NetLin.

NetConv model has extra two convolution layers than NetFull and it uses the ReLU activation function. 

The convolution layer allows the layer to learn a subset of the images each time instead of the whole input compared to the fully connected layer. This can reduce the number of parameters used and allows the model to learn different local features faster. Thus, NetConv performs better than all other two models during the 10 epoch training.

***b) Confusion matrix analysis***

It can be seen that 60000 data are used for training and 10000 are used for testing, with 1000 test data for each class.

|Model|Most mistaken character|Mistaken as|Number of times mistaken|
| :-: | :-: | :-: | :-: |
|1 fully connected layer|4-“na”|2-“su”|76|
|2 fully connected layer|8–“re”|3-“tsu”|62|
|2 convolution layers|2-“su”|3-“tsu”|37|
Most likely mistaken letter is distinguished by seeing the smallest number in the diagonal of the confusion matrix.

|2-“su”|![P87C2T3#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.004.png)|
| - | - |
|3-“tsu”|![P90C4T3#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.005.png)|
|4-“na”|![P93C6T3#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.006.png)|
|8–“re”|![P96C8T3#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.007.png)|

The reason why “na” being mistaken as “su” could be: features such as the tail of these two letters are quite alike, where they all make a curly turn at the bottom of the letter. The overall shape looks like curving towards one direction and looks quite straight.

“re” being mistaken as “tsu” because they both have a shape of a half-closed circle at the tail of the letter, which seems to be a dominant feature for both.

“su” being mistaken as “tsu” may because of the reason aforementioned, where the tail of the letter curve towards one direction.

Overall, “su” and “tsu” are the most frequently mistaken and mistaken as letters due to the reason summarised above.



# **PART 2 - Rectangular Spirals Task**
# Question 1 - code
See code
# Question 2 - code
See code
# Question 3 – show result picture (1 hidden layer)
Appendix c has some tuning process

![P113#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.008.png)

Output:

![P115#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.009.png)

Process:

![P117#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.010.png)![P117#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.011.png)![P117#yIS3](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.012.png)![P117#yIS4](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.010.png)![P117#yIS5](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.013.png)![P117#yIS6](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.014.png)![P117#yIS7](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.015.png)![P117#yIS8](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.016.png)![P117#yIS9](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.017.png)![P117#yIS10](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.018.png)![P117#yIS11](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.019.png)![P117#yIS12](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.020.png)![P117#yIS13](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.021.png)![P117#yIS14](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.022.png)![P117#yIS15](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.023.png)![P117#yIS16](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.024.png)
# Question 4 - show result picture (2 hidden layers)

Layer6 –98.17 – local minimum

Layer7 –98.17 – local minimum

Layer8 --100 –global minimum

Layer10 --100 –global minimum

Output

![P125#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.025.png)

Process

![P127#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.026.png)![P127#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.027.png)![P127#yIS3](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.028.png)![P127#yIS4](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.029.png)![P127#yIS5](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.030.png)![P127#yIS6](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.031.png)![P127#yIS7](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.032.png)![P127#yIS8](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.027.png)
# Question 5 – result discussion

a)

The functions computed by the 1-layer model are all linear functions and the decision boundary is simply a straight line. As for the 2-layer model, it forms a non-linear decision boundary. By looking over all functions made from the 1-layer model, each decision boundary is responsible for different depths (from the outermost data to those in the centre) of the squared spiral data. This also applies to the 2-layer model. It seems that each plot of the hidden nodes takes care of different regions of the data and the final output aggregate them together.

b)

|2 layers|1 layer|
| - | - |
|![P136C3T4#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.033.png)|![P137C4T4#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.034.png)|

From the above comparison, the qualitative difference between the overall function is that the 2 layers model forms a much smoother and accurate decision boundary which oscillates less around the data point, compared to the 1-layer model. For example, the central part of the decision boundary in 1 layer diagram is a lot closer to the opposite class data points and it is not the case for the 2-layer model.





# **PART 3 - Encoder Networks**

|\*||||||||||\*|
| - | - | - | - | - | - | - | - | - | - | - |
|||||||\*|||||
||||\*|||\*|\*||||
|||\*||\*|\*|||\*|||
||\*||||||||\*||
||\*||||||||\*||
|||\*||\*|\*|||\*|||
||||\*|||\*|\*||||
|\*||||||||||\*|
||||||||\*||||
|\*||||||||||\*|
*Table 1* 
![P279TB2#y1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.035.png)![P279#y1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.036.png)





![P285#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.037.png)











||1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|20|
| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
|1|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|0|
|2|1|1|1|1|1|1|1|1|1|1|0|0|0|0|0|0|0|0|0|0|
|3|1|1|1|1|1|1|0|0|0|0|1|0|0|0|0|0|0|0|0|0|
|4|1|1|1|0|0|0|0|0|0|0|1|1|0|0|0|0|0|0|0|0|
|5|1|1|1|1|1|1|0|0|0|0|1|1|0|0|0|0|0|0|0|0|
|6|1|1|1|1|1|1|1|0|0|0|1|1|0|0|0|0|0|0|0|0|
|7|1|1|0|0|0|0|0|0|0|0|1|1|1|0|0|0|0|0|0|0|
|8|1|1|1|1|0|0|0|0|0|0|1|1|1|0|0|0|0|0|0|0|
|9|1|1|1|1|1|0|0|0|0|0|1|1|1|0|0|0|0|0|0|0|
|10|1|1|1|1|1|1|1|1|0|0|1|1|1|0|0|0|0|0|0|0|
|11|1|0|0|0|0|0|0|0|0|0|1|1|1|1|0|0|0|0|0|0|
|12|1|1|1|1|1|1|1|1|1|0|1|1|1|1|0|0|0|0|0|0|
|13|1|0|0|0|0|0|0|0|0|0|1|1|1|1|1|0|0|0|0|0|
|14|1|1|1|1|1|1|1|1|1|0|1|1|1|1|1|0|0|0|0|0|
|15|1|1|0|0|0|0|0|0|0|0|1|1|1|1|1|1|0|0|0|0|
|16|1|1|1|1|0|0|0|0|0|0|1|1|1|1|1|1|0|0|0|0|
|17|1|1|1|1|1|0|0|0|0|0|1|1|1|1|1|1|0|0|0|0|
|18|1|1|1|1|1|1|1|1|0|0|1|1|1|1|1|1|0|0|0|0|
|19|1|1|1|0|0|0|0|0|0|0|1|1|1|1|1|1|1|0|0|0|
|20|1|1|1|1|1|1|0|0|0|0|1|1|1|1|1|1|1|0|0|0|
|21|1|1|1|1|1|1|1|0|0|0|1|1|1|1|1|1|1|0|0|0|
|22|0|0|0|0|0|0|0|0|0|0|1|1|1|1|1|1|1|1|0|0|
|23|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|0|0|
|24|1|1|1|1|1|1|1|0|0|0|1|1|1|1|1|1|1|1|1|0|
|25|0|0|0|0|0|0|0|0|0|0|1|1|1|1|1|1|1|1|1|1|
|26|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|1|
*Table 2**

The approach is based on the geometric property of the autoencoder network. As a result of one-hot encoding, each decision boundary separates the current dot from all other dots. The original image is transformed to Table 1 based on the vertical and horizontal decision boundary. 

In **table 1**, 1 ~ 10 corresponds to the vertical decision boundary and 11~20 for the horizontal line. 

In **table 2**, each column 1~ 20 corresponds to the vertical and horizontal decision boundary line and each row 1 ~ 26 is relative to each data point shown in table 1.

**Table 2** is formed based on the **rules** below:

The dot that is on the left side of the vertical or the top of a horizontal decision boundary is treated as 0. 1 otherwise.

To speed up filling the number, we could do row by row and copy and paste for the same row.




# **PART 4 - Hidden Unit Dynamics for Recurrent Networks**

# Question 1 – SRN, illustrate hidden state
\1) SRN and mapping from hidden unit dynamic to finite state machine

|1|[-0.9996,  0.9995]|
| :-: | :-: |
|2|[ 0.8845, -0.7489]|
|3|[ 0.1856, -0.9893]|
|4|[ 0.9980,  0.4504]|
|5|[ 0.1712,  0.9998]|
|6|[-0.6199, -0.3793]|
*Table 3 – obtain by printing out the “hidden” variable and “state” variable and match them*

![P924#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.038.png) ![P924#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.039.png)
# Question 2 - an bn

The first image below is as required

![P928#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.040.png) ![P928#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.041.png)







# Question 3 - result discussion
![P937#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.042.png) ![P937#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.043.png)

The state trajectory is shown in the above figure. Each of the dots represents an input symbol (a1 ~ a8, b1 ~b8, a8b8) and the cross indicates the initial “a” in the input sequence. 

The left cluster (x < 0.5) represents a’s and the right cluster indicates b’s.  A potential dividing line/decision boundary could be x=0.5.

During each input letter of a’s and b’s, it can be observed from both images that:

with the increase of input a’s, the hidden unit move to the left (-0.2119 > -0.7495 > …), and the increase in b’s causes it to move to the right (0.9829 < 0.9985 < …). 

The network “learns” by counting up the a’s and b’s and adjusting corresponding weight so that the input move towards the two fixed point – centroid of each cluster. This allows the network to learn the pattern/grammar of the sequence and so it can predict the following up sequence given a sequence.

For **how it predicts the last B**, for example, if the a’s the network counts up to is 4, then after 3 consecutive b’s, the network would predict the next one to be b, after that, it finds the counting of B’s is same as A’s which is 4, and **so predict the following A.** (since the network learnt the pattern anbn)

Hidden unit dynamic is also captured by the tensor image above, where the a’s and b’s can be distinguished by the first column sign, where positive is a and negative is b. 



# Question 4 - an bn cn

|Front view|Top |Bottom|Left|Right|
| - | - | - | - | - |
|![P955C6T8#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.044.png)|![P956C7T8#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.045.png)|![P957C8T8#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.046.png)|![P958C9T8#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.047.png)|![P959C10T8#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.048.png)|
![P961#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.049.png)
# Question 5 – result discussion
![P963#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.050.png)![P963#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.051.png)![P963#yIS3](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.052.png)

*Figure 4 – image a) is from the  (Chalup, 2003), the second one is my output, the third is the hidden unit value*

The second image in the above figure is rotated so that its axis matches the first image.

The third image in the above figure shows that A is distinguished by its weight of 1 in the first hidden unit (first column), B is the second and C is the third. This matches with the distribution in the first two images above, where A cluster is closer to the **right side** of the cube and  B is close to the **left** and C is close to the **top**. 

The network (here for anbncn) firstly counts the number of a’s as it converges to the right side of the cube, the same thing to b and c. And then it adjusts the weight such that each distinct letter in the high dimension space is closer to its corresponding hidden unit side. The network, with its three hidden units, successfully learn the anbncn Reber Grammar problem by projecting these input sequences into higher dimension space (here d = 3). Since they are linearly separable in the 3-dimension space, the following sequence can be predicted by each state transition (achieved by counting the a, b, c).

**For example**, if the network counts 4 A’s and 3 B’s, then it will predict the next symbol to be B. Then, it starts to predict C and continue until 4 C’s is counted. Next, it will then predict A since the network learns the pattern that the same number of A’s, B’s, C’s should appear consecutive in the sequence.
# Question 6 – LSTM, analysis behaviour  and explain

![P971#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.053.png)

![P972#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.054.png)

![P973#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.055.png)![P973#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.056.png)

To predict the last symbol (**T** or **P**) in an ERG string, the LSTM needs to remember the second input symbol but not to confuse it with the same symbol in the Reber Grammar box (in the image above).

The structure of LSTM solves the long-term dependency issue aforementioned. The cell state holds the long-term memory. The working memory is in a hidden state. The forget gate forgets the unimportant information. The input gate determines how much input to be used for the cell state.

**In this context**, the cell state could hold memory on the structure of embedded Reber Grammar, which is the second input symbol T or P, follow by some normal Reber grammar and then follow by another T or P. Once the normal Reber grammar pattern is learnt, the forget gate would probably forget the repetitive information for the next input and instead focus on changes such as the second T or P. The input gate will “consolidate” these memories and the updated information is merged into current cell state. This process is repeated until all pattern is learnt. 

![P977#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.057.png)

*Figure 12 from the paper “Knowledge extraction from the learning of sequences in a long-short term memory (LSTM) architecture”*


The above figure interprets how LSTM learns the ERG by extracting different states using k-means clustering in the hidden unit space with k = 10. This illustrates LSTM learns the pattern of the grammar by projecting the ERG sequence input to a higher dimensional space (each dimension relative to a letter in the pre-define grammar) and creating a linear combination of different dimensions such that different ERG instances can be **separable** and thus classified.

Below is based on [LSTM Explorer (echen.me)](http://blog.echen.me/lstm-explorer/#/network?file=counter), an example of an X bn

The hidden state of one neuron, as below image (dark red[-1] to dark blue[1] since tanh), illustrates how the weight is adjusted in LSTM. The symbol closer to the endpoint X is relatively more important (with a more negative weight - darker). The same pattern applies to cell state. **The symbol X** is given with a distinguishable weight in the cell, hidden and forget state. Besides, there also exist neurons that put more weight on the beginning sequence of each symbol. The aggregation of these neurons output allows the LSTM to “remember” the symbol that appears early in the input since the cell state could carry information about previous input and propagate along the learning process.


![P982#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.058.png)![P982#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.059.png)![P982#yIS3](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.060.png)

![P983#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.061.png)![P983#yIS2](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.062.png)![P983#yIS3](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.063.png)

*Figure 2 – neuron 2 in the above demo link*

\-----------------------------------------------------------------------------------------------------------------

Find these paper discusses what seems to be the question here:

“Knowledge extraction from the learning of sequences in a long-short term memory (LSTM) architecture”

[Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples](https://arxiv.org/abs/1711.09576).

Below is a demo that converts LSTM hidden state to finite-state machine

<https://colab.research.google.com/drive/1tkJK1rJVEg9e-QcWOxErDb3cQq9UR-yR#scrollTo=b1OQw8PwS15S>

Reference

For part4 hidden unit dynamic explanation 

Blair, A. D., & Pollack, J. B. (1997). [Analysis of dynamical recognizers](https://www.mitpressjournals.org/doi/pdfplus/10.1162/neco.1997.9.5.1127). *Neural Computation*, *9*(5), 1127–1142.

Chalup, S. K., & Blair, A. D. (2003). [Incremental training of first-order recurrent neural networks to predict a context-sensitive language](https://www.sciencedirect.com/science/article/pii/S0893608003000546). *Neural Networks*, *16*(7), 955–972.






# APPENDIX
**A**

**Number of hidden nodes = 50**

Test set: Average loss: 0.6058, Accuracy: 8120/10000 (81%)

**Number of hidden nodes = 100**

Test set: Average loss: 0.5363, Accuracy: 8376/10000 (84%)

**Number of hidden nodes = 200**

Test set: Average loss: 0.4945, Accuracy: 8468/10000 (85%)

**Number of hidden nodes = 500**

Test set: Average loss: 0.5017, Accuracy: 8471/10000 (85%)

**Number of hidden nodes = 784**

Test set: Average loss: 0.5016, Accuracy: 8466/10000 (85%)

**B**

Below is the result without the fully connected layer (forget), no padding, kernel size for filter = 5, other as default

**1 maxpool** – 88%

**No maxpool** – 90%

**No maxpool + He weight initialisation** – 90% (converge earlier than No maxpool)

Below all with He weight initialisation, other as default

**lr = 0.001** – 84%

**lr = 0.01(default)** – 88%

**mom=0.3** – 90%

**mom=0.7** – 90%

**mom=0.8** – 88%

**mom=0.9** – 90%

**mom=0.95** – 88%

Below is the result with the fully connected layer, no padding, others as default

**1 maxpool** – 92%

**He weight initialisation + mom=0.7** – 94%

**He weight initialisation + mom=0.9** – 94%




**C**


![P1036#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.064.png)

![P1038#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.065.png)

![P1039#yIS1](Aspose.Words.fad72d1d-1554-42e7-a034-e14afdfc4c39.066.png)

And more not listed. The general trend is that if 100% accuracy does not at least once during the first 2000 epoch, then it is very likely that the model could not learn it.

Increase the learning rate or decrease the initial weight seems to be the general solution to the local minimum.
