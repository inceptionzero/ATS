Platform: Java 6

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
