# Kaggle Predict Future Sales Competition

Code used for competition submission

## Overview

The goal of this Kaggle competition is to predict future sales for a large Russian software firm. The dataset has three main dimensions: 

* `shop_id`
* `item_id`
* `date_block_num` (month number)

The train data contains _daily_ observations over 34 months. The aim is to predict `item_count_month`(i.e. total monthly sales) for each `shop_id`/`item_id` pair over the next month. Only a few features are readily available (such as item price, item category, item name or feature name). Model performance is evaluated using RMSE.

## Approach

The main steps of my solution for this competition are outlined below.

### Data Preparation

Little data cleaning was necessary. A few suspicious outliers were spotted for the price and sales variables. These observations were simply removed from the dataset since they represented an unsignificant fraction of the total number of observations. Negative prices were replaced by the median price of the item over the month. 

The most critical step was to correclty aggregate the train data in order to match the structure of the test data. The first step is to aggregate the test data from daily to monthly observations. Then, the train data is augmented such that each possible `shop_id`/`item_id` pair for a given month is present (setting the number of sales to zero for combinations that were not originally present). This step has an important impact on the performance of the model. 

Thanks to the Kaggle forums, I found out that some features could be extracted from the shop and item category names (such as the city or the item type and subtypes). This was not obvious to me since all names are in Russian. I have also created features based on prices (e.g. difference compared to long term trend, 'on sale' flags, difference in price compared to other shops, etc.). The task included many categorical variables, hence mean-encoding proved to be important. Since I was planning to use tree-based models, I also applied mean-encoding over some interactions (tree-based models have a hard time extracting such dependencies on their own). Relevant variables were lagged up to 12 months. 

### Train - Validation Split

The test set runs over the month following the last month of the train set (no overlap). This was reflected in in the train/validation split: I picked the last month of the train set as my validation set. 

### Models

I have tried a few different models for this task:

* **Gradient Boosted Decision Trees (GBDTs)**: GBDTs are known to perform well on this type of application and are widely used in Kaggle competitions. They indeed performed better than the other models I tried. I used both the `XGBoost` and `LightGBM` implementations. Both yielded similar performance (but trees ended up being quite different, based on feature importance). 

* **Random Forest (RF)**: RF didn't perform as well as GBDTs but it was still decent. I thought that the RF model would be able to add value when ensembling but the predictions ended up being too correlated with the ones of the GBDTs. 

* **Ridge Regression**: The main issue here is that many important features are categorical ones with a large number of different categories (e.g. `shop_id` and `item_id`). This is an issue in the regression setting since categorical need to be one-hot encoded. I tried using only the numerical features (including mean-encoded features), along with some scaling transformations but the performance was very poor. I think one could make ridge regression work in this setting but it would require more work (maybe binning some variables would help).

For the tree-based models, I used the `hyperopt` library for hyperparamter tuning. 

### Ensembling

To gain a few performance points, I made an ensemble using the predictions of the two GBDT models. The final prediction is a simple convex combination of two (the weight was optimized on the validation set).

## Further Potential Improvements 

* create more features (e.g. moving averages, binning, etc.)
* use stacking instead of a convex combination for ensembling
* try neural nets