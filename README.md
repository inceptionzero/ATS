Email: sinasheikh.13@gmail.com

Platform: Java

Machine Learning Library: Weka

== Adaptive Trading System (ATS) ==

'''Goal of the system:'''

The objective of the system is to provide a recommendation on the purchase, sale, or retention of generic securities on a single day based on a prediction as whether the next day's closing value of a specific market index would be respectively higher than, lower than, or equal to that of the present day.

----

'''Trading model and assumptions:'''

The system recommends a trading decision based on a single market index, specifically Dow Jones Industrial Average (DJI). The trading interval is a single day. In specific, in the case of a 'purchase' recommendation, a security is purchased on the day of the recommendation and sold on the following day; in the same way, in the case of a 'sale' recommendation, a security is sold on the day of the recommendation and purchased on the following day; in the case of a 'retain' recommendation, no trading is performed. The trading model does not include short sales. The system uses the historical closing value of DJI in determining its trading decision.  

----

'''Trading techniques:'''

The system performs data mining to ''classify'' the next day's closing value of DJI as higher than, lower than, or equal to that of the present day by extracting a pattern from the historical DJI closing value data using a set of technical analysis techniques as data ''attributes''. In specific, simple moving average (SMA), exponential moving average (EMA), volume simple moving average (VSMA), double crossover, moving average convergence-divergence (MACD), and relative strength index (RSI) account for the trending and mean-reversion techniques used as data attributes.

Dataset sample:
@relation dji
@attribute SMA {rising,falling,flat}
@attribute EMA {rising,falling,flat}
@attribute VSMA {rising,falling,flat}
@attribute DoubleCrossOver {GoldenCross,DeadCross,NoCrossOver}
@attribute MACD {positive,negative,centerline}
@attribute RSI {overbought,oversold,neither}
@attribute classIndex {purchase,sale,retain}
@data
falling,falling,rising,NoCrossOver,negative,oversold,retain
falling,rising,rising,GoldenCross,negative,oversold,purchase
rising,rising,falling,NoCrossOver,negative,neither,retain
rising,rising,falling,DeadCross,positive,overbought,sale

----

'''Evaluation metric:'''

Cross validation between the training and test instances is performed on a complete data set using an untrained classifier. The percentage of correctly classified instances and kappa statistic are used as two evaluation metrics.  

kappa = (P(A) - P(E)) / (1 - P(E)), where P(A) is the percentage agreement between the classifier and ground truth, and P(E) is chance agreement. k=1 indicates perfect agreement, and k=0 indicates chance agreement.

----

'''Machine learning algorithm and platform:'''

The following classifiers are used as the system's machine learning algorithms:

Decision Tree classifiers,

(A) C4.5 decision tree

(B) NBTree, a decision tree with Naive Bayes classifiers at the leaves.

(C) RandomForest, a random forest of 10 decision trees, each constructed while considering 3 random features.

Bayes classifiers,

(D) BayesNet, a Bayesian Network classifier.

(E) NaiveBayes, a Naive Bayes classifier.

Rules classifiers,

(F) DecisionTable

(G) ZeroR

(H) Ridor, Ripple Down Rule Learner

Meta Classifier,

(I) AdaBoostM1  

The system is developed in Java using Weka as an external machine learning library.

Note: Please refer to the Appendix for the definition of the machine learning algorithms.

----

'''Data source:'''

Historical closing value record of DJI for the past 6 months, 1 year, 5 years, 10 years, 25 years, 50 years, 70 years, and 82 years are obatined from Yahoo Finance as the source data.

----

'''Data usage:'''

Example-wise randomization is performed on the source data set. Seventy percent of the randomized data set is used as the training set and the remaining thirty percent as the test set.

----

'''Error function:'''

Root mean squared error is used as the error function in evaluating each machine learning experiment.  

----

'''Results:'''

Classification performance of the system (in this case, with C4.5 decision tree as the ML algorithm) decreases as the size of the dataset increases. Figure 1.

[[Image:fig 2'.jpg]]

'' Figure 1. C4.5Tree performance (82-year dataset), Blue: % Correctly classified instances, Red: % Incorrectly classified instances.''


Classification performance of the system decreases, RMSE becomes larger, as the size of the dataset increases. Figure 2.

[[Image:fig 3.jpg]]

'' Figure 2. C4.5Tree performance (82-year dataset), Root Mean Squared Error.''


Classification performance of the system decreases, kappa decreases becomes smaller, as the size of the dataset increases. Figure 3.

[[Image:fig 4.jpg]]

'' Figure 3. C4.5Tree performance (82-year dataset), kappa = (P(A) - P(E)) / (1 - P(E)), where P(A) is the percentage agreement between the classifier and ground truth, and P(E) is chance agreement. k=1 indicates perfect agreement, and k=0 indicates chance agreement.''


Classification performance of all ML algorithms decreases as the size of the dataset increases. On the other hand, the variability between the performance of different ML algorithms decreases as the dataset becomes larger. In other words, the ML algorithms perform more uniformly as the size of the dataset increases. Figure 4.

[[Image:fig 6.jpg]]

'' Figure 4. % Correctly classified instances vs. Duration of historical closing value set (size of dataset)''

''Dark Blue: C4.5Tree''

''Orange: NBTree''

''Yellow: RandomForest''

''Dark Green: BayesNet''

''Brown: NaiveBayes''

''Light Blue: DecisionTable''

''Black: ZeroR''

''Light Green: Ridor''

''Purple: AdaBoostM1''

----

'''Review of the system:'''

The initial implementation of the system resulted in classification performance comparable to chance prediction. As a result, a review was conducted to determine whether (i) a technical problem exists in the implementation of the system or (ii) the set of technical analysis techniques used as dataset attributes failed to provide a statistically significant characterization of dataset instances.

No technical problems were found in the implementation of the system. The generation of training and test datasets were traced and verified at each step of the code. Sanity check was performed to verify the correct configuration of the ML algorithms using datasets for which the calssification results could be verified by hand. Specifically, all ML algorithms scored 100% in correctly classifying instances of datasets in which all instances had their ClassIndex attribute set to one identical nominal value, for example "retain".

The attributes (SMA, EMA, Double Crossover, MACD, and RSI) previously used to define each instance were all derived from the closing DJI value. Volume Simple Moving Average (VSMA), derived from the numbers of shares traded on each day, is also added to the set of attributes used to define each instance. Nevertheless, the new set of attributes (SMA, EMA, VSMA, Double Crossover, MACD, and RSI) still did not uniquely characterize a statistically significant number of instances to improve the classification performance of the system beyond predicition by chance.

As an alternate approach, in the second implementation of the system, a relationship between the attribute set and the ClassIndex was '''introduced''' to the dataset. Previsouly, the ClassIndex of each instance was set to "purchase", "sale", or "retain" exclusively based on whether the next day's closing value of DJI index was higher than, lower than, or the same as that of the present day. However, in order to introduce a relationship between the attribute set and the ClassIndex, the ClassIndex value of each instance was set not only based on the next day's closing value of DJI index compared to that of the present day, but, at the same time, based on the value of one or more technical analysis attributes (i.e. SMA, EMA, VSMA, DCO, MACD, and RSI) of that instance. Figure 1 illustrates the relationship between the specific technical analysis attributes used in setting the ClassIndex value of dataset instances and the reulting classification performance of the system. For all ML algorithms used in this experiment, the highest classificaiton performance (the brown bar) was achieved when the ClassIndex value was set according to the following conditions:

if ((Next day's closing value of DJI > Present day's closing value of DJI) &&
(EMA = "rising" && VSMA = "rising") ||
(Next day's closing value of DJI > Present day's closing value of DJI) &&
(DCO = "GoldenCross" && VSMA = "rising")) ClassIndex = "purchase"

else if ((Next day's closing value of DJI < Present day's closing value of DJI) &&
(EMA = "falling" && VSMA = "falling") ||
(Next day's closing value of DJI < Present day's closing value of DJI) &&
(DCO = "DeadCross" && VSMA = "falling") ClassIndex = "sale"

else ClassIndex = "retain"         

[[Image:fig 1.jpg]]

'' Figure 5. AttributeSet used in Setting ClassIndex vs %Correctly Classified Instances, 82-year dataset''

''Blue: EMA+VSMA, MACD+VSMA, DCO+VSMA, RSI+VSMA''

''Orange: EMA+VSMA, MACD+VSMA, RSI+VSMA''

''Yellow: EMA+VSMA, MACD+VSMA, DCO+VSMA''

''Green: EMA+VSMA, DCO+VSMA, RSI+VSMA''

''Brown: EMA+VSMA, DCO+VSMA''



''C4.5 Decision tree internal models:''

[[Image:C4_5_none_v1.jpg]]

''Figure 6A. Internal model of the C4.5 Decision Tree classifier learned/built upon the completion of a training session, ClassIndex set exclusively based on closing index value ('''no''' relationship between ClassIndex and the attribute set is introduced), % Correctly classified instances ~ 50, 1-year dataset.''

[[Image:C4_5_brown_v2.jpg]]

''Figure 6B. Internal model of the C4.5 Decision Tree classifier learned/built upon the completion of a training session, ClassIndex set simultaneously based on closing index value '''and''' the 'EMA+VSMA && DCO+VSMA' attribute values at each instance, % Correctly classified instances > 80, 1-year dataset.''

Note: The numerical value at each node, refers to the normalized information gain (difference in entropy) achieved as a result of splitting the dataset based on the corresponding attribute. Appendix.


''Comparison of ML-based and Non ML-based trading results:''

With the exception of DCO and RSI, technical analysis techniques result in lower classification performance compared to the ML algorithms. The reason for the fairly high classification performance of DCO and RSI is that a large number of instances within a dataset correspond to the nominal values "NoCrossOver" and "neither" respectively, for which the ClassIndex value is "retain." As a result, although these two technical analysis techniques score fairly high in correctly classifying instances, their performance does not result in higher return on investment as in the majority of instances their trading recommendation is to "retain" and not trade the security.  

[[Image:fig one.jpg]]

'' Figure 7A. % Correctly Classified Instances, 10-year dataset''


Compared to three technical analysis techniques, the C4.5 decision tree machine learning approach yields a higher investment return.

[[Image:fig one'.jpg]]

'' Figure 7B. Amount of money made through each approach, 10-year dataset''

''MACD, $940.86''

''RSI, $1139.64''

''DCO, $536.17''

''C4.5, $60712.81''

----

'''Technical analysis techniques:'''

'''SMA:'''

[[Image:SMA2.jpg]]

''Figure 9. 5-Day Simple moving average, 1-year dataset.''


'''EMA:'''

[[Image:EMA2.jpg]]

''Figure 10. 35-Day Exponential moving average, 1-year dataset.''


'''VSMA:'''

[[Image:VSMA.jpg]]

''Figure 11. 5-Day Volume simple moving average, 1-year dataset.''



'''DCO:'''

[[Image:DCO'.jpg]]

''Figure 12. Double crossover (Golden cross SMA > EMA, Dead cross SMA < EMA), Blue 5-Day SMA, Red 35-Day EMA, 1-year dataset.''


'''MACD:'''

[[Image:MACD'.jpg]]

''Figure 13. 12-Day - 26-Day Moving average convergence divergence, 1-year dataset.''


'''RSI:'''

[[Image:RSI'.jpg]]

''Figure 14. 14-Day Relative strength index (Overbought RSI > 70%, Oversold RSI < 30%), 1-year dataset.''

== Appendix ==

'''C4.5 Decision Tree:'''

The C4.5 algorithm builds decision trees from a set of training data using the concept of information entropy. The training data is a set S = s1,s2,... of already classified samples. Each sample si = x1,x2,... is a vector where x1,x2,... represent attributes of the sample. The training data is augmented with a vector C = c1,c2,... where c1,c2,... represent the class to which each sample belongs. At each node of the tree, C4.5 chooses one attribute of the data that most effectively splits its set of samples into subsets enriched in one class or the other. Its criterion is the normalized information gain (difference in entropy) that results from choosing an attribute for splitting the data. The attribute with the highest normalized information gain is chosen to make the decision. The C4.5 algorithm then recurs on the smaller sublists.

The algorithm has the following base cases, (A) All the samples in the list belong to the same class. When this happens, it simply creates a leaf node for the decision tree saying to choose that class. (B) None of the features provide any information gain. In this case, C4.5 creates a decision node higher up the tree using the expected value of the class. (C) Instance of previously-unseen class encountered. Again, C4.5 creates a decision node higher up the tree using the expected value.

C4.5 pseudocode:

1. Check for base cases.
2. For each attribute ''a'', find the normalized information gain from splitting on ''a''.
3. Let ''a_best'' be the attribute with the highest normalized information gain.
4. Create a decision node that splits ''on a_best''.
5. Recur on the sublists obtained by splitting on ''a_best'', and add those nodes as children of node.

Reference - Wikipedia contributors. "C4.5 algorithm." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 13 Jun. 2010. Web. 15 Jun. 2010.

----

'''NBTree:'''

Input: A set T of labelled instances.

Output: A decision-tree with Naive-Bayes classifiers at the leaves.

NBTree pseudocode:

1. For each attribute Xi, evaluate the utility, U(Xi), of a split on Xi. For continuous attributes, a threshold is also found at this stage.
2. Let j = arg MAXi(Ui), the attribute with the highest utility.
3. If Uj is not significantly better than the utility of the current node, create a Naive-Bayes classifier for the current node and return.
4. Partition T according to the test on Xj. If Xj is continuous, a threshold split is used; if Xj is discrete, a multi-way split is made for all possible values.
5. For each child, call the algorithm recursively on the portion of T that matches the test leading to the child.

Note: The utility of a node is computed by discretizing the data and determining the 5-fold cross-validation accuracy estimate of using Naive-Bayes at the node. The utility of a split is the weighted sum of the utility of the nodes, where the weight given to a node is proportional to the number of instances that go down to that node.

Reference - Ron Kohavi. "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid." Second International Conference on Knoledge Discovery and Data Mining. 1996. [http://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf Manuscript]

----

'''RandomForest:'''

Random forests are a combination of tree predictors such that each tree depends on the values of a random vector sampled independently and with the same distribution for all trees in the forest. The generalization error for forests converges a.s. to a limit as the number of trees in the forest becomes large. The generalization error of a forest of tree classifiers depends on the strength of the individual trees in the forest and the correlation between them. Using a random selection of features to split each node yields error rates that compare favorably to Adaboost (Y. Freund & R. Schapire, Machine Learning: Proceedings of the Thirteenth International conference, &ast;&ast;&ast;, 148–156), but are more robust with respect to noise. Internal estimates monitor error, strength, and correlation and these are used to show the response to increasing the number of features used in the splitting. Internal estimates are also used to measure variable importance. These ideas are also applicable to regression.

Each tree is constructed using the following algorithm:

1. Let the number of training cases be N, and the number of variables in the classifier be M.
2. We are told the number m of input variables to be used to determine the decision at a node of the tree; m should be much less than M.
3. Choose a training set for this tree by choosing N times with replacement from all N available training cases (i.e. take a bootstrap sample).
Use the rest of the cases to estimate the error of the tree, by predicting their classes.
4. For each node of the tree, randomly choose m variables on which to base the decision at that node. Calculate the best split based on these m variables in the training set.
5. Each tree is fully grown and not pruned (as may be done in constructing a normal tree classifier).

''The advantages of random forest:''

For many data sets, it produces a highly accurate classifier.
It handles a very large number of input variables.
It estimates the importance of variables in determining classification.
It generates an internal unbiased estimate of the generalization error as the forest building progresses.
It includes a good method for estimating missing data and maintains accuracy when a large proportion of the data are missing.
It provides an experimental way to detect variable interactions.
It can balance error in class population unbalanced data sets.
It computes proximities between cases, useful for clustering, detecting outliers, and (by scaling) visualizing the data.
Using the above, it can be extended to unlabeled data, leading to unsupervised clustering, outlier detection and data views.
Learning is fast.

''The disadvantages of random forest:''

Random forests are prone to overfitting for some datasets. This is even more pronounced in noisy classification/regression tasks.
Random forests do not handle large numbers of irrelevant features as well as ensembles of entropy-reducing decision trees.
It is more efficient to select a random decision boundary than an entropy-reducing decision boundary, thus making larger ensembles more
feasible. Although this may seem to be an advantage at first, it has the effect of shifting the computation from training time to evaluation
time, which is actually a disadvantage for most applications.

References - 1. Wikimedia: "Random Forest." [http://wapedia.mobi/en/Random_forests Wikimedia] 2. Leo Breiman. "Random Forest." Machine Learning archive, Volume 45 , Pages: 5-32. [http://portal.acm.org/citation.cfm?id=570182 Manuscript]

----

'''BayesNet:'''

A Bayesian network is a probabilistic graphical model that represents a set of random variables and their conditional independencies via a directed acyclic graph (DAG). They are DAGs whose nodes represent random variables in the Bayesian sense: they may be observable quantities, latent variables, unknown parameters or hypotheses. Edges represent conditional dependencies; nodes which are not connected represent variables which are conditionally independent of each other. Each node is associated with a probability function that takes as input a particular set of values for the node's parent variables and gives the probability of the variable represented by the node.

Let U = {x1, . . . , xn}, n ≥ 1 be a set of variables. A Bayesian network B over a set of variables U is a network structure BS, which is a directed acyclic graph (DAG) over U and a set of probability tables BP = {p(u|pa(u))|u ∈ U} where pa(u) is the set of parents of u in BS.
A Bayesian network represents a probability distributions P(U) = Qu∈U p(u|pa(u)).

The classification task consist of classifying a variable y = x0 called the class variable given a set of variables x = x1 . . . xn, called attribute variables. A classifier h : x → y is a function that maps an instance of x to a value of y. The classifier is learned from a dataset D consisting of samples over (x, y). The learning task consists of finding an appropriate Bayesian network given a data set D over U.

References -
1. Remco R. Bouckaert. "Bayesian Networks." Weka Manual. 2009. [http://sourceforge.net/projects/weka/files/documentation/WekaManual-3-7-0.pdf/download Manual]
2. David Heckerman. "A Tutorial on Learning With Bayesian Networks." Microsoft Research. 1996. [http://research.microsoft.com/pubs/69588/tr-95-06.pdf Manuscript]
3. Wikipedia contributors. "Bayesian network." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 3 Jun. 2010. Web. 15 Jun. 2010.

----

'''NaiveBayes:'''

A Bayes classifier is a simple probabilistic classifier based on applying Bayes' theorem (from Bayesian statistics) with strong (naive) independence assumptions. A more descriptive term for the underlying probability model would be "independent feature model". In simple terms, a naive Bayes classifier assumes that the presence (or absence) of a particular feature of a class is unrelated to the presence (or absence) of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 4" in diameter. Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier considers all of these properties to independently contribute to the probability that this fruit is an apple. Depending on the precise nature of the probability model, naive Bayes classifiers can be trained very efficiently in a supervised learning setting. In many practical applications, parameter estimation for naive Bayes models uses the method of maximum likelihood; in other words, one can work with the naive Bayes model without believing in Bayesian probability or using any Bayesian methods. In spite of their naive design and apparently over-simplified assumptions, naive Bayes classifiers have worked quite well in many complex real-world situations. In 2004, analysis of the Bayesian classification problem has shown that there are some theoretical reasons for the apparently unreasonable efficacy of naive Bayes classifiers. Still, a comprehensive comparison with other classification methods in 2006 showed that Bayes classification is outperformed by more current approaches, such as boosted trees or random forests. An advantage of the naive Bayes classifier is that it requires a small amount of training data to estimate the parameters (means and variances of the variables) necessary for classification. Because independent variables are assumed, only the variances of the variables for each class need to be determined and not the entire covariance matrix.

References - 1. Wikipedia contributors. "Naive Bayes classifier." Wikipedia, The Free Encyclopedia. Wikipedia, The Free Encyclopedia, 16 Sep. 2010. Web. 28 Sep. 2010. 2. Harry Zhang. "The Optimality of Naive Bayes". FLAIRS2004 conference. [http://www.cs.unb.ca/profs/hzhang/publications/FLAIRS04ZhangH.pdf Manuscript] 3. Caruana, R. and Niculescu-Mizil, A.: "An empirical comparison of supervised learning algorithms". Proceedings of the 23rd international conference on Machine learning, 2006.

----

'''Decision table:'''

A decision table is typically divided into four quadrants: Conditions, Condition alternatives, Actions, and Action entries.

Each decision corresponds to a variable, relation or predicate whose possible values are listed among the condition alternatives. Each action is a procedure or operation to perform, and the entries specify whether (or in what order) the action is to be performed for the set of condition alternatives the entry corresponds to. Many decision tables include in their condition alternatives the don't care symbol, a hyphen. Using don't cares can simplify decision tables, especially when a given condition has little influence on the actions to be performed. In some cases, entire conditions thought to be important initially are found to be irrelevant when none of the conditions influence which actions are performed.

Aside from the basic four quadrant structure, decision tables vary widely in the way the condition alternatives and action entries are represented. Some decision tables use simple true/false values to represent the alternatives to a condition (akin to if-then-else), other tables may use numbered alternatives (akin to switch-case), and some tables even use fuzzy logic or probabilistic representations for condition alternatives[citation needed]. In a similar way, action entries can simply represent whether an action is to be performed (check the actions to perform), or in more advanced decision tables, the sequencing of actions to perform (number the actions to perform).

Decision table can be used if the combination of conditions are given. In decision table conditions are known as causes and serinal numbers of conditions are known as business rule.

Reference - Wikipedia contributors, "Decision table," Wikipedia, The Free Encyclopedia, http://en.wikipedia.org/w/index.php?title=Decision_table&oldid=389542180 (accessed October 23, 2010).

----

'''ZeroR:'''

ZeroR is a learner used to test the results of the other learners. ZeroR chooses the most common category all the time. ZeroR learners are used to compare the results of the other learners to determine if they are useful or not, especially in the presence of one large dominating category. ZeroR classifier utilizes 0-R classifier(s) and predicts the mean (for a numeric class) or the mode (for a nominal class).

Reference - [http://www.google.com/url?sa=t&source=web&cd=10&sqi=2&ved=0CE4QFjAJ&url=http%3A%2F%2Fciteseerx.ist.psu.edu%2Fviewdoc%2Fdownload%3Fdoi%3D10.1.1.143.4974%26rep%3Drep1%26type%3Dpdf&rct=j&q=zeror%20classifier&ei=X3TCTPO5JIO0sAOSs937Cw&usg=AFQjCNFBvaW91M8_5ofV_gigrNoaIgHEjg&sig2=fhNWFlOXPvNxJ0AY0twijw Manuscript]

----

'''Ridor:'''

Ridor is an implementation of a Ripple Down Rules (RDR) learner. It generates a default rule first and then the exceptions for the default rule with the least (weighted) error rate. Then it generates the "best" exceptions for each exception and iterates until pure. Thus it performs a tree-like expansion of exceptions.The exceptions are a set of rules that predict classes other than the default. IREP is used to generate the exceptions.

RDR consist of a data structure and knowledge acquisition scenarios. Human experts' knowledge is stored in the data structure. The knowledge is coded as a set of rules. The process of transferring human experts's knowledge to Knowledge Based Systems in RDR is explained in Knowledge Acquisition Scenario.

Data Structure: There are various structures of Ripple Down Rules, for example Single Classification Ripple Down Rules (SCRDR), Multiple Classification Ripple Down Rules (MCRDR), Nested Ripple Down Rules (NRDR) and Repeat Inference Multiple Classification Ripple Down Rules (RIMCRDR). The data structure of RDR described here is SCRDR, which is the simplest structure. The data structure is similar to a decision tree. Each node has a rule, the format of this rule is IF cond1 AND cond2 AND ... AND condN THEN conclusion. Cond1 is a condition (boolean evaluation), for example A=1, isGreater(A,5) and average(A,">",average(B)). Each node has exactly two successor nodes, these successor nodes are connected to predecessor node by "ELSE" or "EXCEPT". An example of SCRDR tree (defined recursively) is shown below:

IF (OutLook = "SUNNY" AND Temperature = "COOL") THEN PLAY="TENNIS" EXCEPT Child-1 ELSE Child-2, where Child-1 and Child-2 are also SCRDR trees.
For example Child-1 is: IF (Wind = "WINDY" AND Humidity = "HIGH") THEN Play="SQUASH" EXCEPT NoChild ELSE NoChild

Knowledge Acquisition Scenario: Human experts provide a case to the systems and they add a new rule to correct the classification of a misclassified case. For example rule Child-1 is added to correct classification of case [OutLook="SUNNY", Temperature="COOL", Wind="WINDY", Humidity="HIGH", ForeCast="STORM", Play="SQUASH"]. This case is misclassified as Play="TENNIS". When a rule is constructed by the human experts, the conditions of this rule should be satisfied by the misclassified case and also they should NOT be satisfied by any previous cases classified correctly by the parent rule (in this context is the first rule).

Reference - Wikipedia contributors, "Ripple down rules," Wikipedia, The Free Encyclopedia, http://en.wikipedia.org/w/index.php?title=Ripple_down_rules&oldid=387270680 (accessed October 23, 2010).

----

'''AdaBoostM1:'''

Boosting is a general method for improving the accuracy of any given learning algorithm. This short paper introduces the boosting algorithm AdaBoost, and explains the underlying theory of boosting, including an explanation of why boosting often does not suffer from overfitting. Some examples of recent applications of boosting are also described.

References -
1. Robert E. Schapire. "A brief introduction to boosting." International Joint Conference On Artificial Intelligence archive, Proceedings of the 16th international joint conference on Artificial intelligence,  Volume 2, Pages: 1401-1406 [http://portal.acm.org/citation.cfm?id=1624417 Manuscript]
