# Proposal

The goal of our project is to predict the cost of an airbnb listing given its public listing information, which can ultimately be used either to estimate how much a new listing should be put on the market for or to predict whether an existing listing is over/underpriced. 
Reference works include [Li et Al’s price prediction for specific properties based off proximity to landmarks and facilities](http://www.public.asu.edu/~swang187/publications/Price_Recommendation.pdf) and [Tang and Sangani’s neighborhood and price range prediction for particular neighborhoods](http://cs229.stanford.edu/proj2015/236_report.pdf).
For the datasets, we’ll be using a combination of the data available at [Inside Airbnb](http://www.insideairbnb.com) and [2010 US Census data](https://www.census.gov) provided by zipcode. The data at inside Airbnb provides all the public information on all the airbnb listings by certain cities, including availability information and price fluctuation by day of the year. The census data will be useful to compare general trends in housing prices per zip code so that our model can handle predicting prices for cities with very different housing price ranges (i.e. Boston vs. Austin). 
Regarding technical details, we’re planning on using tf-idf and potentially PCA in the process of featurization and a regression algorithm for classification, most likely linear regression to actually predict a price for a listing and not just a over/underpriced label. As far as analysis, we’ll rely on mean squared error to evaluate the success of our model.

At the moment, we are planning to pair on all tasks until we discover the best way to split up the featurization and classification tasks.

## Goals:
* **April 20th:** featurization
* **MVP for May 15th:** predict price for a listing in 1 to a few cities (i.e. Boston/SF/Austin) and, if it’s an existing listing, give an opinion on whether it’s over/underpriced.
* **The ideal final outcome:** predict prices for any US city with a more generalized model
* ***Stretch goal:*** website or chrome extension that allows you to determine whether an existing airbnb listing is over/underpriced.
