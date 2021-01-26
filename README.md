# Classification Of Inappropriate Comments On Social Networks Using Text Mining
## Introduction
The Technological advancement and the current pandemic the Covid -
19 is leading millions of people to spend their time on Social networks,
like Facebook, Twitter, Instagram, YouTube, etc. The networks give the possibility that every person in the world
can express themselves through photos videos and comments. It seems
that in every social network there is a place to leave a comment as you
like. my goal is to reduce the number of inappropriate words over the internet.


## What I Needed First

1. to get the data:    
find a dataset of comments from an online video named “Donald Glover's This is America” from YouTube.

2. to get the words:  
find a dictionary that combines a lot of
offensive words and swear-words. We will tag our dataset using a 
substring searching algorithm to determine harmful words. 

## Pre-Processing
• Convert bad-words.txt file to list.  
• Import comments.csv using python’s pandas library to get the
data as DataFrame.  
• Change 'textOriginal' column name to 'comment'.  
• Remove Nulls  
• Tag each row as appropriate/inappropriate using a substring
detection algorithm. For each comment, if it has a harmful word,
tag it by 1, else, 0.  
• Create transformed csv with two columns: comment[str] and
inappropriate[binary]  
• Count vectorized each text.  
• Remove stop words (I, am, is and etc.)  
• balance the data - reduced the
number of comments for the acceptable
category.


## The Algorithms
for the classification problem, I implement machine learning algorithms:  
1. Naïve Bayes
2. Support Vector Machine
3. K-neighbors

## The Results
Confiusin Matrix for each algorithm  
  
![LinearSVC](https://i.ibb.co/zVj6jwC/1212.png)    

![final](https://i.ibb.co/p46v0wV/Whats-App-Image-2021-01-23-at-14-07-36-1.jpg)    

  
For the highest true positive is SVM, the value is 1. meaning for
comments on social media, the chance that the algorithm will be
perceived as an unacceptable comment is 100%, that Indicates a
problem with the algorithm since it is not possible to have an
accuracy of 100%. although, the algorithm doesn’t have the highest
ability to detect an acceptable comment. We can notice it from the
confusion matrix of SVM.
Because of all of that the SVM isn’t the right algorithm for the
project, he is not serving our essence.
The chosen algorithm by the largest area under the curve and the
highest ROC is NB. The ability to classify text as
unacceptable/acceptable is 89% with 14% of errors.

## So What Next?
In order to reach the maximum of the project, by strengthen and
train the algorithm we recommend collecting more sources of
information, so that the algorithm will also learn different forms of
comments besides the comments of YouTube, and adding more
words to a dictionary to verify and validate each comment in the best
way and better tune and configure the model.

## Credits
[data.world](https://data.world/popculture/donald-glovers-this-is-america-youtube-comments) for the dataset.  
to the [Dictonary](https://github.com/RobertJGabriel/Google-profanity-words/blob/master/list.txt) of inappropriate words.   
to the great tutorial for [Text Classification with Scikit-Learn](https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f)
