
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

## Data Cleaning



