# Data Collection

## Datasets:

We'll be primarily using Airbnb data from 3 cities with various cost of living ranges: 
  - Asheville, North Carolina (http://data.insideairbnb.com/united-states/nc/asheville/2016-04-18/data/listings.csv.gz)
  - Austin, Texas (http://data.insideairbnb.com/united-states/tx/austin/2015-11-07/data/listings.csv.gz)
  - Boston, Massachusetts (http://data.insideairbnb.com/united-states/ma/boston/2016-09-07/data/listings.csv.gz)
  
Eventually, we might increase our dataset to include other cities with different housing price values, but we have determined these three cities as a good range to start with.

### Training Dataset


### Development Dataset


### Testing Dataset


### Housing Cost Dataset
Used to account for differences in cost of livings in different zip codes

Link to download and/or view data: https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_15_5YR_B25075&prodType=table

Source: U.S. Census Bureau, 2011-2015 American Community Survey 5-Year Estimates
Universe: Owner-occupied housing units

## Method

- clean up airbnb listing dataset by dropping unecessary cols (and the ones that don't occur in all 3 cities)
  ['id', 'listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped', 'weekly_price', 'monthly_price', 'neighbourhood_cleansed', 'license', 'jurisdiction_names']
  
 - sort it out, 10% of each city's listing data into dev and test, the other 80% into training
 
 - certain text columns only have a few unique "default" values (like host_response_time) so we can turn those into integer values vs. featurizing them like text
 
 
 cancellation_policy: ['flexible' 'strict' 'moderate' 'super_strict_30']
 host_response_time:  ['within an hour' 'within a day' nan 'within a few hours' 'a few days or more']
 amenities
 
 
 
 
