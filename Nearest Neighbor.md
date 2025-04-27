Let's find cheese that meticulously well paired with my meat.
What makes one cheese better than another? 
It turns out that all cheeses can be categorized by a few distinct characteristics.
The different characteristics of a cheese are known as features.
Cheese that share similar characteristics will be our definition of the nearest neighbor.

In the ML world, information about each cheese would be referred to as the data.
We are going to need a large amount of data about cheese. So we will index cheese.com.
This is considered factual information about cheese but also includes many opinionated discussions about cheese.
All this data together will be a wealth of information for making decisions.

A cheese expert would consider all the characteristics together to classify a given cheese.
A nearest neighbor algorithm does something similar but in a natural language processing (NLP) way.
It compares words (or phrases) from different cheeses against one another. Depending on how similar they are, a probability is returned.
It will be a probability that the two cheeses are a good fit as a number.

If you’ve ever attempted text comparisons on a large dataset you will know that it’s anything but performant.
To overcome this, the text is converted to a collection of numbers called vectors.
The act of converting text to numerics is known as tokenization.

Anything in the database that has vectors similar to those words is probably a neighbor - complementing cheeses.
Finding the nearest neighbor is the process of plotting all the vectors in all their dimensions and then comparing a context collection of vectors to them.
Using a simple coordinate system you can mathematically measure how far one point is from another (known as their distance).


K-nearest neighbors (KNN)
Classify some piece of data against a large set of labeled data
Label: what each item in the data set is
Unsupervised data -> Labeling -> Supervised data
The “K” in KNN is a representation of bounds: meaning how many pictures of cheese are you willing to consider?  how many points in space you are willing to consider?
a prediction of how well the provided data fits the existing data label => a percentage and a classifier


Approximate Nearest Neighbor (ANN)
it works well on non-labeled (unsupervised) data
The return will be approximately what data is closely related to the input, but hallucinations are real so be careful.


Fixed radius nearest neighbor
Fixed radius is an extended approach to KNN. it is limited to a certain distance.
Limiting the number of points to consider is an easy way to speed up the overall calculation.
The context value is a vector and the radius(fixed value) is a measure of distance from that vector.


Partitioning with k-dimensional tree(k-d tree)
When data is tokenized (converted to vectors) the number of dimensions is chosen. based on how accurate you need a search to be
The more dimensions an embedding has the longer it’s going to take to compute a nearest neighbor. Need balance
k-d tree splits the single space into a number of spaces (called partitions)
How the spaces are sorted (so their shared context is not lost) is an implementation choice (median-finding sort).
the number of leaves that make up each space in the tree is balanced. This makes for uniform, predictable search performance.
the algorithm is given a sense of proximity.
it can choose to not search large portions of the tree because it knows those leaves are too far. That can really speed up a search.
