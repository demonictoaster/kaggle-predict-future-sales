# Kaggle Predict Future Sales Competition

Code used for competition submission

## Overview

The goal of this Kaggle competition is to predict future sales for a large Russian software firm. The dataset has three main dimensions: 

* `shop_id`
* `item_id`
* `date_block_num` (month number)

The train data contains _daily_ observations over 34 months. The aim is to predict `item_count_month`(i.e. total monthly sales) for each `shop_id`/`item_id` pair over the next month. Only a few features are readily available (such as item price, item category, item name or feature name). 

## Approach

The main steps of my solution for this competition are outlined below.

### Data Preparation

Little data cleaning was necessary. A few suspicious outliers were spotted for the price and sales variables. These observations were simply removed from the dataset since they represented an unsignificant fraction of the total number of observations. Negative prices were replaced by the median price of the item over the month. 

## Result

## Further Potential Improvements 