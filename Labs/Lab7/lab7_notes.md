# Lab 7 Notes

## 1 Introduction
Machine learning is now ubiquitous in computer vision, meaning most real-world applications use machine learning in some capacity. In this lab we look at some of these practices to gain insight into how computer vision is done on an industrial level.

## 2 Training and Testing Algorithms
The program called *ml* is able to train a system on a variety of vision tasks and test them, generating an output that can be display using FACT.
A sample command could be *./ml -learner=svm train recog.kb train/\*"*, which tells it to train, saving the result in the file *recog.kb* using the data files located in the train folder. The last argument can also be a task file to circumvent overflow which often is the case when training on a large sample of images.
The -learner qualifier tells the program which support vector machine to use for the training, the possibiliies being: 
* cnn: Convolutional Neural Network
* eigen: The “eigenfaces” algorithm described in Chapter 8 of the notes. My implementation of this is particularly simplistic, which means it yields overly poor results and takes ages to run: O(N2) rather than O(N).
* mlp: Multi-Layer Perceptron
* rf: Random Forests — not described in the lecture notes but quite widely used in practice
* svm: Support Vector Machine
* wisard: WISARD

## 3 Assessing Algorithms' Performance Individually
In this step we try to rank the results of the various machine learning algorithms using their result files and FACT. Below are the scores of each result.

### EIGEN
Error rates calculated from mnist-eigen.res
   tests      TP      TN      FP      FN accuracy   recall precision specificity class
     980     980       0       0       0     1.00     1.00      1.00        0.00 0
    1135       0       0    1135       0     0.00     0.00      0.00        0.00 1
    1032       0       0    1032       0     0.00     0.00      0.00        0.00 2
    1010       0       0    1010       0     0.00     0.00      0.00        0.00 3
     982       0       0     982       0     0.00     0.00      0.00        0.00 4
     892       0       0     892       0     0.00     0.00      0.00        0.00 5
     958       0       0     958       0     0.00     0.00      0.00        0.00 6
    1028       0       0    1028       0     0.00     0.00      0.00        0.00 7
     974       0       0     974       0     0.00     0.00      0.00        0.00 8
    1009       0       0    1009       0     0.00     0.00      0.00        0.00 9
   10000     980       0    9020       0     0.10     1.00      0.10        0.00 overall

Confusion matrix calculated from mnist-eigen.res
                                	expected
actual      0      1      2      3      4      5      6      7      8      9
     0    980   1135   1032   1010    982    892    958   1028    974   1009

### MLP
Error rates calculated from mnist-mlp.res
   tests      TP      TN      FP      FN accuracy   recall precision specificity class
     980     967       0      13       0     0.99     1.00      0.99        0.00 0
    1135    1123       0      12       0     0.99     1.00      0.99        0.00 1
    1032    1002       0      30       0     0.97     1.00      0.97        0.00 2
    1010     990       0      20       0     0.98     1.00      0.98        0.00 3
     982     956       0      26       0     0.97     1.00      0.97        0.00 4
     892     864       0      28       0     0.97     1.00      0.97        0.00 5
     958     935       0      23       0     0.98     1.00      0.98        0.00 6
    1028     994       0      34       0     0.97     1.00      0.97        0.00 7
     974     938       0      36       0     0.96     1.00      0.96        0.00 8
    1009     984       0      25       0     0.98     1.00      0.98        0.00 9
   10000    9753       0     247       0     0.98     1.00      0.98        0.00 overall

Confusion matrix calculated from mnist-mlp.res
                                    expected
actual      0      1      2      3      4      5      6      7      8      9
     0    967      0      3      0      0      1      5      0      7      1
	 1      1   1123      2      1      0      0      3      5      1      2
     2      3      3   1002      4      4      0      4     11      1      0
	 3      0      1      4    990      0     11      1      6      5      5
     4      0      0      3      0    956      0      3      1      4      7
 	 5      2      1      2      7      2    864      5      1      4      2
     6      1      2      3      0      5      5    935      0      5      0
	 7      2      1      6      5      1      2      0    994      4      5
	 8      4      4      7      2      2      6      2      2    938      3
     9      0      0      0      1     12      3      0      8      5    984

### RF
Error rates calculated from mnist-rf.res
   tests      TP      TN      FP      FN accuracy   recall precision specificity class
     980     969       0      11       0     0.99     1.00      0.99        0.00 0
    1135    1123       0      12       0     0.99     1.00      0.99        0.00 1
    1032     999       0      33       0     0.97     1.00      0.97        0.00 2
    1010     977       0      33       0     0.97     1.00      0.97        0.00 3
     982     956       0      26       0     0.97     1.00      0.97        0.00 4
     892     861       0      31       0     0.97     1.00      0.97        0.00 5
     958     941       0      17       0     0.98     1.00      0.98        0.00 6
    1028     989       0      39       0     0.96     1.00      0.96        0.00 7
     974     931       0      43       0     0.96     1.00      0.96        0.00 8
    1009     961       0      48       0     0.95     1.00      0.95        0.00 9
   10000    9707       0     293       0     0.97     1.00      0.97        0.00 overall

Confusion matrix calculated from mnist-rf.res
                                       expected
   actual      0      1      2      3      4      5      6      7      8      9
        0    969      0      7      0      1      2      5      1      5      5
        1      0   1123      0      0      0      0      3      2      0      4
        2      0      2    999      8      2      1      0     19      5      2
        3      0      3      6    977      0     10      0      1      8     10
        4      0      0      3      0    956      3      3      0      3     12
        5      2      2      0      6      0    861      4      0      5      1
        6      3      2      3      0      4      5    941      0      3      1
        7      1      1      8      8      0      1      0    989      4      5
        8      4      2      6      9      2      6      2      3    931      8
        9      1      0      0      2     17      3      0     13     10    961

### SVM
Error rates calculated from mnist-svm.res
   tests      TP      TN      FP      FN accuracy   recall precision specificity class
     980     972       0       8       0     0.99     1.00      0.99        0.00 0
    1135    1126       0       9       0     0.99     1.00      0.99        0.00 1
    1032    1013       0      19       0     0.98     1.00      0.98        0.00 2
    1010     993       0      17       0     0.98     1.00      0.98        0.00 3
     982     963       0      19       0     0.98     1.00      0.98        0.00 4
     892     867       0      25       0     0.97     1.00      0.97        0.00 5
     958     944       0      14       0     0.99     1.00      0.99        0.00 6
    1028     997       0      31       0     0.97     1.00      0.97        0.00 7
     974     949       0      25       0     0.97     1.00      0.97        0.00 8
    1009     973       0      36       0     0.96     1.00      0.96        0.00 9
   10000    9797       0     203       0     0.98     1.00      0.98        0.00 overall

Confusion matrix calculated from mnist-svm.res
                                       expected
   actual      0      1      2      3      4      5      6      7      8      9
        0    972      0      5      0      0      3      5      1      3      3
        1      0   1126      1      0      0      0      2      7      0      3
        2      1      3   1013      2      5      0      1     10      2      1
        3      0      1      0    993      0     10      0      2      6      7
        4      0      0      1      0    963      1      2      2      5     10
        5      3      1      0      2      0    867      3      0      2      1
        6      1      1      1      0      3      4    944      0      2      1
        7      1      1      7      6      0      1      0    997      2      7
        8      2      2      3      5      1      4      1      1    949      3
        9      0      0      1      2     10      2      0      8      3    973

### WISARD
Error rates calculated from mnist-wisard.res
   tests      TP      TN      FP      FN accuracy   recall precision specificity class
     980      85       0       0     895     0.09     0.09      1.00        0.00 0
    1135       0       0       0    1135     0.00     0.00      0.00        0.00 1
    1032      24       0       4    1004     0.02     0.02      0.86        0.00 2
    1010       7       0       4     999     0.01     0.01      0.64        0.00 3
     982       1       0       1     980     0.00     0.00      0.50        0.00 4
     892       1       0       2     889     0.00     0.00      0.33        0.00 5
     958       3       0       4     951     0.00     0.00      0.43        0.00 6
    1028       0       0       0    1028     0.00     0.00      0.00        0.00 7
     974       0       0       7     967     0.00     0.00      0.00        0.00 8
    1009       0       0       2    1007     0.00     0.00      0.00        0.00 9
       0       0       0       0       0     0.00     0.00      0.00        0.00 failure
   10000     121       0      24    9855     0.01     0.01      0.83        0.00 overall

Confusion matrix calculated from mnist-wisard.res
                                                   expected
   actual      0      1      2      3      4      5      6      7      8      9
        0     85      0      3      0      1      0      3      0      1      0
        2      0      0     24      2      0      0      1      0      3      1
        3      0      0      0      7      0      2      0      0      2      0
        4      0      0      0      0      1      0      0      0      0      0
        5      0      0      1      1      0      1      0      0      1      0
        6      0      0      0      0      0      0      3      0      0      0
        8      0      0      0      0      0      0      0      0      0      1
        9      0      0      0      1      0      0      0      0      0      0
  failure    895   1135   1004    999    980    889    951   1028    967   1007

#### Ranking Based on Accuracy
1. SVM
2. MLP
3. RF
4. EIGEN
5. WISARD

#### Ranking Based on Confusion Matrix
1. SVM
2. MLP
3. RF
4. EIGEN
5. WISARD

## 4 Comparing the Performance of Algorithms
Comparison of mnist-eigen.res and mnist-mlp.res
  Z-score  class    better
     3.33  0        mnist-eigen.res
    33.48  1        mnist-mlp.res
    31.62  2        mnist-mlp.res
    31.43  3        mnist-mlp.res
    30.89  4        mnist-mlp.res
    29.36  5        mnist-mlp.res
    30.55  6        mnist-mlp.res
    31.50  7        mnist-mlp.res
    30.59  8        mnist-mlp.res
    31.34  9        mnist-mlp.res
     0.00  failure  neither

Comparison of mnist-eigen.res and mnist-rf.res
  Z-score  class    better
     3.02  0        mnist-eigen.res
    33.48  1        mnist-rf.res
    31.58  2        mnist-rf.res
    31.23  3        mnist-rf.res
    30.89  4        mnist-rf.res
    29.31  5        mnist-rf.res
    30.64  6        mnist-rf.res
    31.42  7        mnist-rf.res
    30.48  8        mnist-rf.res
    30.97  9        mnist-rf.res
     0.00  failure  neither

Comparison of mnist-eigen.res and mnist-svm.res
  Z-score  class    better
     2.47  0        mnist-eigen.res
    33.53  1        mnist-svm.res
    31.80  2        mnist-svm.res
    31.48  3        mnist-svm.res
    31.00  4        mnist-svm.res
    29.41  5        mnist-svm.res
    30.69  6        mnist-svm.res
    31.54  7        mnist-svm.res
    30.77  8        mnist-svm.res
    31.16  9        mnist-svm.res
     0.00  failure  neither

Comparison of mnist-eigen.res and mnist-wisard.res
  Z-score  class    better
    29.88  0        mnist-eigen.res
     0.00  1        neither
     4.69  2        mnist-wisard.res
     2.27  3        mnist-wisard.res
     0.00  4        neither
     0.00  5        neither
     1.15  6        mnist-wisard.res
     0.00  7        neither
     0.00  8        neither
     0.00  9        neither
     0.00  failure  neither

Comparison of mnist-mlp.res and mnist-rf.res
  Z-score  class    better
     0.35  0        mnist-rf.res
     0.00  1        neither
     0.37  2        mnist-mlp.res
     2.31  3        mnist-mlp.res
     0.00  4        neither
     0.42  5        mnist-mlp.res
     1.18  6        mnist-rf.res
     0.62  7        mnist-mlp.res
     1.04  8        mnist-mlp.res
     3.35  9        mnist-mlp.res
     0.00  failure  neither

Comparison of mnist-mlp.res and mnist-svm.res
  Z-score  class    better
     1.51  0        mnist-svm.res
     1.15  1        mnist-svm.res
     2.29  2        mnist-svm.res
     0.52  3        mnist-svm.res
     1.38  4        mnist-svm.res
     0.49  5        mnist-svm.res
     1.94  6        mnist-svm.res
     0.38  7        mnist-svm.res
     2.18  8        mnist-svm.res
     1.92  9        mnist-mlp.res
     0.00  failure  neither

Comparison of mnist-mlp.res and mnist-wisard.res
  Z-score  class    better
    29.66  0        mnist-mlp.res
    33.48  1        mnist-mlp.res
    31.24  2        mnist-mlp.res
    31.32  3        mnist-mlp.res
    30.87  4        mnist-mlp.res
    29.34  5        mnist-mlp.res
    30.50  6        mnist-mlp.res
    31.50  7        mnist-mlp.res
    30.59  8        mnist-mlp.res
    31.34  9        mnist-mlp.res
     0.00  failure  neither

Comparison of mnist-rf.res and mnist-svm.res
  Z-score  class    better
     0.89  0        mnist-svm.res
     0.76  1        mnist-svm.res
     2.77  2        mnist-svm.res
     2.94  3        mnist-svm.res
     1.55  4        mnist-svm.res
     1.18  5        mnist-svm.res
     0.76  6        mnist-svm.res
     1.32  7        mnist-svm.res
     3.01  8        mnist-svm.res
     2.16  9        mnist-svm.res
     0.00  failure  neither

Comparison of mnist-rf.res and mnist-wisard.res
  Z-score  class    better
    29.70  0        mnist-rf.res
    33.48  1        mnist-rf.res
    31.19  2        mnist-rf.res
    31.11  3        mnist-rf.res
    30.87  4        mnist-rf.res
    29.29  5        mnist-rf.res
    30.59  6        mnist-rf.res
    31.42  7        mnist-rf.res
    30.48  8        mnist-rf.res
    30.97  9        mnist-rf.res
     0.00  failure  neither

Comparison of mnist-svm.res and mnist-wisard.res
  Z-score  class    better
    29.75  0        mnist-svm.res
    33.53  1        mnist-svm.res
    31.42  2        mnist-svm.res
    31.37  3        mnist-svm.res
    30.98  4        mnist-svm.res
    29.39  5        mnist-svm.res
    30.64  6        mnist-svm.res
    31.54  7        mnist-svm.res
    30.77  8        mnist-svm.res
    31.16  9        mnist-svm.res
     0.00  failure  neither

1. SVM
2. MLP
3. RF
4. WISARD
5. EIGEN

## 5 Training and Testing a Face Recognizer
Not doin this ;)
