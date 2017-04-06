import pandas as pd
import csv
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

def drop_airbnb_cols(filename):
    df = pd.read_csv(filename)
    df.drop(['id', 'listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped', 'weekly_price', 'monthly_price', 'neighbourhood_cleansed',              'license', 'jurisdiction_names', 'square_feet', 'neighbourhood', 'calculated_host_listings_count'], axis=1, inplace=True)
    return df

def get_col_names(files):
    counts = {}
    
    for f in files: 
        for c in drop_airbnb_cols(f).columns: 
            if c in counts: 
                counts[c] += 1
            else: 
                counts[c] = 1

    cols = []
    for c in counts: 
        if counts[c] == 3:
            cols.append(c)
    
    return cols

def segment(vector, train, dev, test): 
    count = len(vector)
    test += vector[-1*int(count*.1):]
    dev += vector[-1*int(count*.2):-1*int(count*.1)]
    train += vector[:int(count*.8)]
    
    return train, dev, test
    
def featurize(df):
    
    df['price'] = df['price'].map(lambda x: x.replace('$', "").replace(',',""))
    df[['price']] = df[['price']].apply(pd.to_numeric) # turn the price col into a number col
    cols = df.columns.tolist()
    cols.remove('price')
    cols = ['price'] + cols
    df = df[cols]
    text_cols = df.select_dtypes(exclude=['float64', 'int64'])
#     for col in text_cols: 
#         corpus = df[col].fillna(value="").values
#         print corpus
#         vectorizer = CountVectorizer()
#         X = vectorizer.fit_transform(corpus)
#         print X
    num_cols = df.select_dtypes(include=['float64', 'int64'])
    num_cols.fillna(value=0, inplace=True)
    return [list(i) for i in num_cols.as_matrix()]
    
    # call dictvectorizor on that list --> numpy array of features
    # convert rest of df into numpy array with price as first value
    # append 2 numpy arrays by index
    # return big numpy array

def create_datasets():
    col_names = get_col_names(airbnb_files)
    train = []
    dev = []
    test = []
    for f in airbnb_files:   
        df = pd.read_csv(f, dtype={'zipcode': 'str'})
        df = df[col_names]
        vector = featurize(df)
        train, dev, test = segment(vector, train, dev, test) # put data from f into train, dev, test
    return train, dev, test

def save_datasets(train, dev, test):
    with open('data/train.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(train)

    with open('data/dev.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dev)

    with open('data/test.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test)

if __name__ == '__main__':
    save_datasets(*create_datasets())

