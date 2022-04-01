# Book Reading

Feature engineering
- book by Zheng and Casari (2016).

Word sense semi-supervisr learninh
- automatic discovery of word
senses from text was actually the first place semi-supervised learning was
applied to NLP

traditional NLP
1. Manning, Christopher D., and Hinrich Schütze. (1999). Foundations of
Statistical Natural Language Processing. MIT press.
2. Bird, Steven, Ewan Klein, and Edward Loper. (2009). Natural
Language Processing with Python: Analyzing Text with the Natural
Language Toolkit. O’Reilly.
3. Smith, Noah A. (2011). Linguistic Structure prediction. Morgan and
Claypool.
4. Jurafsky, Dan, and James H. Martin. (2014). Speech and Language
Processing, Vol. 3. Pearson.
5. Russell, Stuart J., and Peter Norvig. (2016). Artificial Intelligence: A
Modern Approach. Pearson.
6. Zheng, Alice, and Casari, Amanda. (2018). Feature Engineering for
Machine Learning: Principles and Techniques for Data Scientists.

misc:
- 123

# Paper Example

- Yelp Review dataset
  - Zhang, Zhao, and Lecun (2015) 

# Coding workflow

## Set seed

```py
seed = 23

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
```

