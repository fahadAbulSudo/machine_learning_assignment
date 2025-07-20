Offline machine learning / batch machine learning:
1.	We have all the data which we need in batches.
2.	We split it into train and test
3.	We train the model, fit it and then predict.
4.	However, if we got another set of data or batch of data then we need to do the split, train and fit steps again and again and we need to build new model each time

Incremental machine learning/ online machine learning:
1.	Data is in stream format. One stream or one phase of data is fed at a time
2.	With each stream of data, our model learns or updates.
3.	online learning is very adaptable because it does not make any assumptions about the distribution of the data
4.	If our data distribution drifts due to say changing customer behavior with time, the model can adapt on-the-fly to keep pace with the trend.
5.	To achieve this in offline learning, we would have to create a sliding window of the data and retrain the model each time.
6.	In a streaming setting, feature scaling is done using running statistics, a data structure that allows the mean and standard deviation to be updated incrementally.

When to use online learning:
1.	Your data does not fit into the memory
2.	You expect the distribution of the data to morph or drift over time
3.	Your data is a function of time (e.g stock prices)

Libraries we can use for online learning:
1.	River: It is a combination of:
a.	Creme
b.	Scikit-Multiflow
2.	MOA - Massive Online Analysis
3.	SAMOA - Scalable Advanced Massive Online Analysis developed by Apache
4.	StreamDB (spark streaming)

River:
1.	Provide list of algorithms for regression, classification and clustering algorithms
2.	examples: SGD, naive bayes, tree-ensemble model, factorization machines, online random forest, ISVM etc
3.	River provide drift detection algorithms.
a.	Concept Drift: In the context of data streams, it is assumed that data can change over time. The change in the relationship between the data (features) and the target to learn is known as Concept Drift
b.	Concept drift can occur in cases like: demand of electricity across the year, stock market, likelihood of new movies.
4.	River provides methods for handling imbalanced data. example: fraud detection, spam classification: RandomUnderSample, RandomOverSample, RandomSampler
5.	In a streaming setting, feature scaling is done using running statistics, a data structure that allows the mean and standard deviation to be updated incrementally
6.	For training a model incrementally, a common learning algorithm is stochastic gradient descent (SGD): The core idea of SGD is that at each training step, the model parameter weights are adjusted in the opposite direction of the gradient, which is computed using the model prediction error at that step.
7.	There is a concept called River pipeline
8.	There are stream generators which feed the data in stream format.

Advantages of Incremental learning:
1.	It does not required large amount of training dataset initially
2.	It can continuously learn to improve when the system is running
3.	It can detect the drift and adapt on-the-fly to the changes

Challenges faced by incremental learning:
1.	The model has to adapt gradually so sometimes it becomes difficult to maintain such models.
2.	Preservation of previously acquired knowledge and without the effect of catastrophic forgetting.
a.	Catastrophic forgetting: The major challenge for incremental learning is catastrophic forgetting, which refers to the drastic performance drop on previous tasks after learning new tasks. This phenomenon is caused by the inaccessibility to previous data while learning on new data
b.	This can be avoided by using rehearsal mechanism: feeding some old information along with new information

