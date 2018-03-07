# Diabetes_analysis
Machine Learning project - 2018

This Machine Learning project is still on going.

The objective of the project is to find correlation, and design a model for diabetes monitoring/detection based on the urine sample.
The motivation of this work is to find an easy, convenient and non-painful (blood test) way to monitor diabetes. This project can
be considered as a part of my master degree consisting of designing and developing an Urine Sensor.

The used data can be found [HERE](https://www.kaggle.com/cdc/national-health-and-nutrition-examination-survey)


# Description
In this project, the data preprocessing was quite challenging (feature's name unintelligible, missing data, cross-checking between the different .csv)
I recommend, like I did to have a look at the actual inquiry and questionnaire ([HERE](https://wwwn.cdc.gov/nchs/nhanes/ContinuousNhanes/Default.aspx?BeginYear=2013))

Firstly, I selected urine data, demographic data and used the questionnaire answer to create my label "Diabetes".
Then I clean the data, and get rid of the feature with too many missing value. Finally, I oversample the diabetes label.
After this step, I got the following data. It is really few features but I still decided to continue the analysis.

<img src="/image/df_framefeature.PNG" width="250">

The objective of the primary analysis was to obtained the importance feature. To do so, I trained to model with random forest and Catboost
And I compare the results:

Random forest: 

<img src="/image/forest_score.PNG" width="250"> <img src="/image/forest_matrix.PNG" width="250"> <img src="/image/forest_ROC.PNG" width="250">

Catboost:

<img src="/image/cat_score.PNG" width="250"> <img src="/image/cat_matrix.PNG" width="250"> <img src="/image/cat_ROC.PNG" width="250">

Feature importance: (right: Random forest, left: Catboost)

<img src="/image/forest_feat.PNG" width="250"> <img src="/image/cat_feat.PNG" width="220"> 

With this result, we can see that it is possible to have an estimation of the diabetes based on the urine sample and simple test. The quality of the model (score) is not excellent and can be increase with more tunning and other data. This Machine Learning project is still on going.

# Code
In this repository, you can find a notebook or python version of the code. 
You will need: Sklearn, Catboost, Seaborn, imblearn modul

Do not hesitat if you have any question :)


