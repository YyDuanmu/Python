# Product Application Fraud Detection

## [Data Quality Report](https://github.com/YyDuanmu/Python/blob/8951561c1d699bb108b4e6afd0a1e44177ee0097/application_fraud/application_DQR.pdf)
The **`Application`** dataset is a dataset of product application with PII (Personal Identifying Information – the fields that identify the person). It covers the whole year of 2016 (01/01/2016 – 12/31/2016) and consists of 10 fields with 1,000,000 records.


**`TABLE FOR NUMERIC FIELDS`**

![image](https://user-images.githubusercontent.com/96048575/169014130-57ab7cba-930f-490c-87b0-6162593d600c.png)


**`TABLE FOR CATEGORICAL FIELDS`**   

![image](https://user-images.githubusercontent.com/96048575/169014251-631a5ba3-c762-46a8-9b9d-909feb57e922.png)


## [Data Cleaning](https://github.com/YyDuanmu/Python/blob/0de91e436e65877d98c484c019eee68ea75d3d12/application_fraud/application_feature_engineering.ipynb)

In the data cleaning section, we did data manipulations after completing exploratory data analysis to handle exclusions, missing values, outliers, and frivolous values.

- **`date`**

In the dataset, the date value was defined as int64, and our approach is to convert them into datetime datatype. We used the pandas to_datetime function with a lambda function to achieve the datatype conversion.

- **`ssn`**

The ssn variable in the dataset refers to the social security numbers of the applicants. As we dig into the dataset, out of 1,000,000 observations, 16,935 observations contain the value of 999999999 which is the frivolous value. If we link them or count them, we can get undesired results, primarily false positives. Therefore, these values need to be addressed. We converted the frivolous ssn values to negative record numbers to make them unique, then added leading zeros to make the length of the field nine digits. In that way, they will not be able to link. We have also added leading zero back for those ssns started with zero.

- **`address`**

The address variable refers to the applicants' addresses. We discovered that 1,079 observations have ‘123 MAIN ST' in address value. With this observation, we concluded that this address is a frivolous value with unusually high frequency. Since record number is a unique identifier, we converted these frivolous addresses into a string and added record number to the end, so that they are unable to link.

- **`zip5`**

The zip5 variable refers to the applications' five-digit zip code values. We noticed that not all the records have five digits, so we set the input to five digits and pad with leading zeros to make them all in the same format.

- **`dob`**

The variable dob refers to the date of birth of the applicant. The dob column has 126,568 observations with the value of 19060626. The value has an unusually high frequency, so we classified it as a frivolous value. We adjusted the values to correlate with the record numbers to unlink by converting frivolous dob values to negative record numbers.

- **`homephone`**

The homephone variable refers to the home phone of the applicant. As we dug into the dataset, we found out that there were 78,512 observations with the value of 9999999999. The value has an unusually high frequency, so we classified it as a frivolous value. Hence, we need to do some data modification to address this issue. The steps are similar to the previous frivolous values we modified. We adjusted the values to correlate with the record numbers to unlink by converting frivolous homephone values to negative record numbers and adding zeros to match the field's length.

- **`age`**

This is the field we created. It is used to record the age of applicants. We calculated the age field using the applicant’s application date minus the date of birth of applicants to finalize the result. In this age field, we also fix the frivolous value of dob 19070626 by converting them to negative record numbers to make them unique.

## [Feature Engineering](https://github.com/YyDuanmu/Python/blob/0de91e436e65877d98c484c019eee68ea75d3d12/application_fraud/application_feature_engineering.ipynb)

### Candidate Variables

An individual fraudster usually goes through a list of victims’ identity information and uses the victims’ core identity information such as SSN, name and DOB and his own contact information such as address and phone number applying for many products. A victim’s identity was usually compromised in a data breach and his core identity information is being used by many fraudsters. Therefore, many applications, if fraudulent, come from the same address and/or phone number; many identities come from the same address and/or phone number; many applications come from the same SSN with many different addresses and/or phone numbers.

Based on the above logic, we created four types of candidate variables: 

-	**`Days since last seen`** indicating how many days since that entity or combination of the entities last appeared in the dataset,
-	**`Velocity`** indicating number of records with that entity or combination of the entities seen over the past n days where n = 0, 1, 3, 7, 14, 30
-	**`Relative velocity`** indicating ratio of the short-term velocity where n = 0, 1 to a longer-term velocity where n = 3, 7, 14, 30, and 
-	**`Unique count`** indicating number of unique counts of particular entity or combination of the entities for another entity or combination of the entities for the past n days where n = 1, 3, 7, 14, 30, 60. 

Additionally, 

**`Risk Table Variables`**
We also created a risk table variable for day of week based on the date of application using data before 2016-11-01 (training data) to transform the categorical field to a numerical field. Specifically, we used the Bayesian method to encode this field and used the average of the fraud_label for all records in each day of week as the numeric representative of the categorical field. We finalized this field by smoothing the value to make sure it smoothly transits between the low number to the high number.


### Final Candidate Variables

![image](https://user-images.githubusercontent.com/96048575/169016607-a9d97114-87bf-4832-82a5-0cb49e4912c0.png)


## [Feature Selection](https://github.com/YyDuanmu/Python/blob/54d9f07b370b05d7cdeca87f6c1e39de36215e20/application_fraud/application_feature_selection.ipynb)

In the feature selection step, we utilized a univariate filter and followed by a wrapper. We stored our candidate variables in three separate files in the step above. We set the balance equal to 0 so that all variables in each of the files will be used. The filter runs separately on each file of variables we created and keeps the top 100 variables from each file. When creating the filter, we removed the last two months as the **`out-of-time`** data, and we also removed the first two weeks of records since the variables are not well formed. We added a continuous random number variable that is uniformly distributed to make sure it does not come up as important when filtering. For each of the variables, we calculated a KS (Kolmogorov-Smirnov) score using pre-assigned fraud labels (goods and bads). After we finished calculating all the scores for each of the candidate variables, we sorted them by descending order and chose the top 100 variables with highest scores. This filter process helped us to get rid of highly correlated variables. 

For the wrapper, we chose to use forward selection and set the number of variables we wanted to keep in the end to be 30. We initially had 2 candidate models in mind: Random Forest and LightGBM. We found that the final average score of variables under the LightGBM model is `0.533`, which is slightly higher than the final average score of 0.523 we got under the Random Forest model. Meanwhile, the LightGBM model is much faster than the Random Forest model. We spent `30 minutes` wrapping around the variables we got from filtering using LightGBM but spent 1 hour and 16 minutes using the Random Forest model. Therefore, we decided to use the LightGBM model as our final model for the wrapper to wrap around the 100 candidate variables we got from the filtering process. 

**`Logic/Steps of Final 5 Variables Selection`**

1.	We used the top 200 variables to train the LightGBM model, checked the feature importance and selected the 50 most important variables to restart training. (All the variables are not similar. For example, we only used homephone_count_3 instead of using homephone_count_30 and homephone_count_7 and other homephone_count_ variables even if their scores or importance are high.
2. 	Then, we selected 5 most frequent basic attributes from top variables with highest feature importance and high scores. Based on this, we chose the variables consisting of above attributes with highest feature importance. (The most frequent basic attributes are fulladdress, homephone, ssn, dob, firstname.) Actually, we will find all these five attributes are specific to find a person accurately.


**`Final Five Variables`**

<img width="937" alt="image" src="https://user-images.githubusercontent.com/96048575/169020798-277c73ff-2ab7-4d53-941c-6f30dd41c755.png">

![image](https://user-images.githubusercontent.com/96048575/169018142-d6aec7d4-63f3-4188-8b8d-8e1d654b0240.png)


## [Model Exploration](https://github.com/YyDuanmu/Python/blob/54d9f07b370b05d7cdeca87f6c1e39de36215e20/application_fraud/application_models.ipynb)

<img width="963" alt="image" src="https://user-images.githubusercontent.com/96048575/169020268-f690bf2d-21d5-4687-a48c-293f6bfa152a.png">


## Results

**Note** that we get rid of the first two weeks' data (38,511 records) since some of them are incomplete. We also filter the last two months out as the OOT data. The leftover data is our train and test data (794,995 records). We then split this data into 70% train data and 30% test data randomly and run the XGBoost model.

***Below is the summary table for the training dataset:***

<img width="851" alt="image" src="https://user-images.githubusercontent.com/96048575/169020624-170a38af-3b43-4748-9e15-123e2225d118.png">

***Below is the summary table for the test dataset:***

<img width="851" alt="image" src="https://user-images.githubusercontent.com/96048575/169020706-0a7fff0b-e101-4ca2-8a08-c2eff7ffd23e.png">


***Below is the summary table for the out-of-time (OOT) dataset:*** 

<img width="851" alt="image" src="https://user-images.githubusercontent.com/96048575/169020721-6206db55-e4a7-476e-a2bf-3be7bd631f2d.png">














