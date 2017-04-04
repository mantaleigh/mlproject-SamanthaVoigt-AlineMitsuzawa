# Data Collection

## Datasets:

We'll be primarily using Airbnb data from 3 cities with various cost of living ranges: 
  - Asheville, North Carolina
  - Austin, Texas
  - Boston, Massachusetts
  
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

- clean up airbnb listing dataset by dropping unecessary cols:
  ['id', 'listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped']
  
 - sort it out, 10% of each city's listing data into dev and test, the other 80% into training
 
 
