# Data Collection

## Datasets:

We'll be primarily using Airbnb data from 3 cities with various cost of living ranges:
  - Asheville, North Carolina (http://data.insideairbnb.com/united-states/nc/asheville/2016-04-18/data/listings.csv.gz)
  - Austin, Texas (http://data.insideairbnb.com/united-states/tx/austin/2015-11-07/data/listings.csv.gz)
  - Boston, Massachusetts (http://data.insideairbnb.com/united-states/ma/boston/2016-09-07/data/listings.csv.gz)

Eventually, we might increase our dataset to include other cities with different housing price values, but we have determined these three cities as a good range to start with.

### Housing Cost Dataset
Used to account for differences in cost of livings in different zip codes.

Link to download and/or view data: https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_15_5YR_B25075&prodType=table

Source: U.S. Census Bureau, 2011-2015 American Community Survey 5-Year Estimates
Universe: Owner-occupied housing units

## Method

### Data Cleaning:
* We cleaned up the Airbnb listing dataset by dropping uneccessary columns (like urls), columns that didn't occur in all 3 cities, and ones that had primarily NaN values in our datasets. For example, we removed:
  * id
  * listing_url
  * license
  * square_feet (primarily NaN)
  * host_id
  * (and more)
  
* Looking Forward:
  * While really digging into the data for this milestone, we realized that we'll probably have to do some additional cleaning of the Airbnb dataset in the future, particularly around NaN values. We might also eventually drop more columns if they prove to be unecessary or problematic.
  
### Dataset Distribution

  * In order to split the data into training, development, and testing datasets, we simply took 80% of the listings from each city and used the aggregated set of all three as testing. The devlopment and testing dataset received the other 20% of the listings from all three cities, evenly distributed between the two.

### Featurizing

* This was the more complicated aspect of our data collection, and ultimately we have only just reached the tip of the iceberg as far as featurization for this project goes.
  
* The Airbnb dataset has (in general) three different types of feature values: 
  * Text descriptions (such as listing description and reviews), which will be featurized using bag-of-words
  * Raw numbers (such as number of bedrooms and number of guests the listing can accomodate), which we have featurized initially by just their raw values
  * Categorical or boolean values (such as if the host is a superhost or host response time), which will be featurized by assigning numerical values to different categories

* Our current plan is to save each of these three different types of features into their own files. For this milestone, we have taken every feature in the raw number category, featurized them, and saved them into corresponding training, development, and testing csv files. In the near future, we will work on featurizing the text descriptions and categorical values. 

* Looking forward, one of the biggest things we have to do is determine what to replace the NaN values with in all the columns. It might be best to change our method from column to column, but currently we're thinking that maybe some combination of using average values or simply using zeros might be sufficient. Right now, we have all NaNs from the raw numbers category set to 0, but we don't think that's the best way long-term. As far as for text descriptions or categorical values, we're a bit more unsure of what the best plan is, since using an empty string as a replacement for NaN is not something we do for our featurizers.

### Census Data Parsing
* The parsing of the census data was simpler. As it stands, we are planning on just using the mean housing value for every zipcode as a scaling factor, so it will not be included in our model. Instead it will be a scaling factor added to our predicted price. Therefore, all we needed to do was to read in the census data, calculate the mean housing value for each zipcode, and save that information into a Python dictionary that we can use for easy lookup later.
