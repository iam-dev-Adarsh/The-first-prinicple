
PS F:\It's_AWESOME!\First_Principles\10.Naive_Bayes> python main.py
============Hand written NaivBayes================
success! 'shaka shaka boom boom' 
 P(spam) = 0.994, P(ham) = 0.006 -> spam
failure! 'maximus decimus meridius' 
 P(spam) = 0.870, P(ham) = 0.130 -> spam
success! 'Baby stand ready for arrival' 
 P(spam) = 0.922, P(ham) = 0.078 -> spam
success! 'project complete review' 
 P(spam) = 0.094, P(ham) = 0.906 -> ham

Accuracy: 3/ 4
=========sklearn MultiNomial NB===============================
1 'shaka shaka boom boom'  P(spam)= 0.996 -> spam
0 'maximus decimus meridius'  P(spam)= 0.864 -> spam
1 'Baby stand ready for arrival'  P(spam)= 0.922 -> spam
1 'project complete review'  P(spam)= 0.091 -> ham