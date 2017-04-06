import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

airbnb_files = ['data/raw_data/Austin_listings.csv', 'data/raw_data/Boston_listings.csv', 'data/raw_data/Asheville_listings.csv']
census_file = 'data/raw_data/housing_val_by_zip.csv'

def parse_zipcode_data(filename):
    '''
    Input: Filename for the census zipcode to housing values data
    Output: A python dictionary mapping zipcodes (as strings) to the mean housing value for that zipcode
    '''

    zipcodes_to_mean_cost = {}
    # median values of the "bins" for the original column ranges
    column_names = ['zipcode', 'total', '5000', '12499.5', '17499.5', '22499.5', '27499.5', \
                    '32499.5', '37499.5', '42499.5','52499.5', '62499.5', '72499.5', '82499.5', \
                    '92499.5', '112499', '137499','162499', '187499', '224999', '274999', \
                    '349999', '449999', '624999', '874999', '1249999', '1749999', '2000000']

    df = pd.read_csv(filename, dtype={'Id2': 'str'})
    df.drop(['Id', 'Geography'], axis=1, inplace=True)
    cols = [c for c in df.columns if c.lower()[:15] != 'margin of error']
    df=df[cols]
    df.columns = column_names

    for index, row in df.iterrows():
        row_sum = 0
        for median_val in df.columns[2:]:
            row_sum += float(median_val)*float(row[median_val])

        if row['total'] > 0:
            zipcodes_to_mean_cost[row['zipcode']] = row_sum/row['total']
        else:
            zipcodes_to_mean_cost[row['zipcode']] = 0

    return zipcodes_to_mean_cost