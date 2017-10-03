# churn
For Kaggle Competition
https://www.kaggle.com/c/kkbox-churn-prediction-challenge 

Given labeled dataset of whether a customer churned, predict whether someone else will churn. 


Stuff we need to do:
  1. Download and preprocess data (I think it's already pre processed)
  2. Figure out how to read in the data in batches. (see next_batch.py for potential option)
  3. Feature Selection
  4. Build out models (Chase = scikit learn & Tensorflow. Gary = ?)
  5. Tune Hyperparameters
  6. Optimize for log loss function
     logloss=−1N∑i=1N(yilog(pi)+(1−yi)log(1−pi))
  
  
