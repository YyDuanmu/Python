
# Credit Card Transaction Fraud Detection
## [Data Quality Report](https://github.com/YyDuanmu/Python/blob/f8a917ebe0486b8617937f69b9f19b8167f342f1/credit_card_transaction_fraud/card_DQR.pdf)
The **`Credit Card Transactions`** Data is a dataset of actual credit card purchases from a US government organization. It covers the whole year of 2006 (2006-01-01 – 2006-12-31), consisting of 10 fields with 96,753 records.

**`TABLE FOR NUMERIC FIELDS`**
| Field Name|	% Populated	| Min |	Max |	Mean	| Stdev |	% Zero |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Date	| 100 | 2006-01-01	| 2006-12-31	| N/A	|N/A|	0|
| Amount	|100|	0.01|	3,102,045.53|	427.89|	10,006.14|	0|

**`TABLE FOR CATEGORICAL FIELDS`**   
|Field Name|	% Populated	|# Unique Values	|Most Common Value|
|:---:|:---:|:---:|:---:|
|Recnum	|100|	96,753|	N/A|
|Cardnum|	100|	1,645|	5142148452|
|Merchnum|	96.51	|13,091	|930090121224|
|Merch description	|100|	13,126|GSA-FSS-ADV|
|Merch state	|98.76|	227	|TN|
|Merch zip	|95.19|	4,567	|38118|
|Transtype|	100|	4|	P|
|Fraud	|100|	2	|0|

## [Data Cleaning](https://github.com/YyDuanmu/Python/blob/eebb662c84cb339db50569feb2a9b5b8f960314f/credit_card_transaction_fraud/cleaning_engineering.ipynb)

I first build the **`cleaning`** function to fill null values in several columns in the following step.

```python
def cleaning(df, i, j):
    data = pd.DataFrame(df[df[j].notnull()], columns=[i, j])
    data = dict(data.groupby(i)[j].apply(lambda x: scipy.stats.mode(x)[0][0]))
    return data
```

The function groups the data by ‘ *i* ’ column, finds mode of ‘ *j* ’ column where the records share the same ‘ *i* ’ column value and fills null values in ‘ *j* ’ column with the **mode**. Briefly, this method aims to fill null values based on given information.

## [Feature Engineering](https://github.com/YyDuanmu/Python/blob/eebb662c84cb339db50569feb2a9b5b8f960314f/credit_card_transaction_fraud/cleaning_engineering.ipynb)

### Basic New Variable - `country`

**`country`**: US, Canada, Foreign, unknown

### Candidate Variables

Credit card transaction fraud often happens in three scenarios. 
- First, someone steals a credit card or finds and uses a lost card. 
- Second, someone at a merchant could “steal” many credit cards by making several counterfeit cards or having skimmers at gas stations, ATMs, and/or other merchants. 
- Third, someone gets the username and password for an online account and therefore gets the credit card information. 

Based on these scenarios, we identify several characteristics of potential transaction fraud including but not limited to the burst of activities, larger than normal purchase amounts, increased usage in card-not-present, activities at new merchants, high-risk merchant or new geography, and fictitious transactions or merchants.

Based on the above logic, we create four types of candidate variables:

- **`Amount`** variables indicate differently calculated amounts by an entity or combination of the entities over the past n days where n = 0, 1, 3, 7, 14, 30 and amount is average, maximum, median, total, actual/average, actual/maximum, actual/median and actual/total
- **`Frequency`** variables indicate the number of transactions with an entity or combination of the entities over the past n days where n = 0, 1, 3, 7, 14, 30
- **`Days-since`** variables indicate how many days since an entity or combination of the entities last appeared in the dataset
- **`Velocity`** change variables indicate the ratio of the short-term frequency where n = 0, 1 to a longer-term frequency where n = 3, 7, 14, 30

Additionally, 

**`Risk Table Variables`**
We also create two risk table variables, one for day of week and one for state, based on the date and state of the transaction respectively using data before 2006-11-01 (training data) to transform the categorical field to a numerical field. Specifically, we use the Bayesian method to encode these two fields and use the average of the Fraud for all records in each day of week/state as the numeric representative of the categorical field. We finalize these two fields by smoothing the value to make sure it smoothly transits between the low number to the high number.

**`Benford’s Law variables`**
Our last type of variable is Benford’s Law variable where we quantify the difference of the first digit distribution for each cardholder and merchant from Benford’s Law Probabilities. Benford’s Law is the observation that the first digit of many measurements is not uniformly distributed and
follows a specific distribution. We use Benford’s Law to detect fraud where a potential fraudster is making up a large amount of numbers.
Since we may not have enough records for every cardholder or merchant to have ten bins, we divided the first digit distribution into two bins, a low bin with 1 and 2 and a high bin with 3 through 9. We then calculate a measure of unusualness U based on Benford’s Law distribution and smoothed the value as U*. We exclude transactions from Fedex in this process since they violate Benford’s Law but are not unusual. The top 40 potential cardholder fraudsters and merchant fraudsters based on Benford’s law are listed in Appendix II for future reference.

### Final Candidate Variables

![image](https://user-images.githubusercontent.com/96048575/168711110-a7c18608-6473-4c56-a922-22af26b88f7a.png)

## [Feature Selection](https://github.com/YyDuanmu/Python/blob/ee1a80d330bf7617f0fcb4adc4d9c56253c91425/credit_card_transaction_fraud/featureselection_model.ipynb)

In the feature selection step, we utilize a univariate filter followed by a wrapper. The filter runs all of candidate variables we created and keeps the top 100 variables. When creating the filter, we remove the last two months’ data as **`out-of-time (OOT)`** data, and we also remove the first two weeks of records since the variables are not well-formed. We add a continuous random number variable that is uniformly
distributed to make sure it does not come up as important when filtering. For each of the variables, we calculate a Kolmogorov-Smirnov (KS) score using pre-assigned fraud labels (goods and bads). After we finish calculating all the scores for each of the candidate variables, we sort them by
descending order and choose the top 100 variables with the highest scores. This filter process helps us to get rid of highly correlated variables. For the wrapper, we choose to use forward selection and set the number of variables we wanted to keep in the end to be 30. We use the LightGBM model as our final model for the wrapper to wrap around the 100 candidate variables we get from the filtering process. We choose to keep the top 30 variables with highest average scores (0.722). This wrapper process helps us to further remove correlated variables and only leave the ones that are truly important and useful for our analysis.



## [Model Exploration](https://github.com/YyDuanmu/Python/blob/ee1a80d330bf7617f0fcb4adc4d9c56253c91425/credit_card_transaction_fraud/featureselection_model.ipynb)

![image](https://user-images.githubusercontent.com/96048575/168712216-79325c66-7548-4499-88a1-af45a326aeed.png)


















