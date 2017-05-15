import pandas as pd
import numpy as np
import scipy.sparse as sp
import math, time, warnings, csv
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore') # not ideal - to fix

airbnb_files = ['data/raw_data/Austin_listings.csv', 'data/raw_data/Asheville_listings.csv' , 'data/raw_data/Boston_listings.csv', 'data/raw_data/Chicago_listings.csv', 'data/raw_data/Neworleans_listings.csv', 'data/raw_data/LA_listings.csv', 'data/raw_data/portland_listings.csv', 'data/raw_data/Nashville_listings.csv']

def parse_zipcode_data(filename):
    '''

    Currently unused - necessary for normalizing using cost of living data.

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
    '''
        input: a filename for detailed airbnb listing data
        output: a pandas dataframe with unecessary columns dropped
    '''

    df = pd.read_csv(filename)
    df.drop(['id', 'listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_id', 'host_url', 'host_thumbnail_url', 'host_picture_url', 'calendar_last_scraped', 'weekly_price', 'monthly_price', 'neighbourhood_cleansed', 'license', 'jurisdiction_names', 'square_feet', 'neighbourhood', 'calculated_host_listings_count', 'first_review', 'last_review', 'country', 'country_code',             'latitude', 'longitude', 'host_name', 'host_location', 'market', 'state', 'city', 'is_location_exact',             'smart_location', 'has_availability', 'calendar_updated', 'host_listings_count', 'experiences_offered', 'host_since', 'requires_license'], axis=1, inplace=True)
    return df

def get_col_names(files):
    '''
        input: a list of detailed airbnb listing data files
        output: a list of common column names (columns that occur in all the files)
    '''

    counts = {}

    for f in files:
        for c in drop_airbnb_cols(f).columns:
            if c in counts:
                counts[c] += 1
            else:
                counts[c] = 1

    cols = []
    for c in counts:
        if counts[c] == len(files):
            cols.append(c)

    return cols

def clean_percents(x):
    try:
        if math.isnan(x):
            return x
    except TypeError:
        return float(x.strip('%'))/100

def clean_prices(x):
    try:
        if math.isnan(x):
            return x
    except TypeError:
        return float(x.replace('$',"").replace(',',""))

def clean_price_col(df, max_price):
    '''
        Clean the price column (get rid of symbols and turn it into a float).
        Returns the modified dataframe.
    '''

    df['price'] = df['price'].map(lambda x: x.replace('$', "").replace(',',""))
    df[['price']] = df[['price']].apply(pd.to_numeric) # turn the price col into a number col
    df = df[df['price'] < max_price]  # gives a warning
    df = df[pd.notnull(df['price'])]

    return df

def featurize_categorical(df, max_price):
    df = clean_price_col(df, max_price)

    # boolean
    bool_cols = ['require_guest_profile_picture', 'require_guest_phone_verification', 'requires_license', 'instant_bookable']
    bool_map = {'t': 1, 'f': 0}
    for col in bool_cols:
        df[col].replace(bool_map, inplace=True)
    # categorical - rn treating everything else as categorical to simplify things
     # TODO: separate ordinal columns
    categorical_cols = ['bed_type', 'cancellation_policy', 'room_type']
    for col in categorical_cols:
        unique_vals = df[col].unique()
        cat_map = { unique_vals[i]: i for i in range(len(unique_vals)) }
        df[col].replace(cat_map, inplace=True) # slow but ok for now
    v = [list(i) for i in df.as_matrix()]
    return v

def featurize_text(df, max_price, type_vec, max_feats):
    df = clean_price_col(df, max_price)

    prices = df['price']
    X = prices.reshape(prices.shape[0], -1)
    for col in df.columns:
        if col != 'price':
            corpus = df[col].fillna(value="").values #np complains bc i'm modifying a view
            if type_vec == 'count':
                vectorizer = CountVectorizer(stop_words='english', max_features=max_feats)
            elif type_vec == 'tfidf':
                vectorizer = TfidfVectorizer(stop_words='english', max_features=max_feats)
            x = vectorizer.fit_transform(corpus) #TODO: clean text
            X = sp.hstack((X, x))
    return X

def featurize_num(df, max_price):
    df = clean_price_col(df, max_price)

    # clean up
    df['host_response_rate'] = df['host_response_rate'].apply(clean_percents)
    df['host_acceptance_rate'] = df['host_acceptance_rate'].apply(clean_percents)
    df['security_deposit'] = df['security_deposit'].apply(clean_prices)
    df['cleaning_fee'] = df['cleaning_fee'].apply(clean_prices)
    df['extra_people'] = df['extra_people'].apply(clean_prices)

    # threshold maximum_nights to 365 if it's over
    df.ix[df.maximum_nights > 365, 'maximum_nights'] = 365

    # change nan's to 0
    to_change_to_0 = ['reviews_per_month', 'beds', 'bedrooms', 'bathrooms', 'host_response_rate', 'host_acceptance_rate', 'review_scores_accuracy', 'review_scores_communication', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_rating', 'review_scores_value', 'review_scores_checkin', 'security_deposit', 'cleaning_fee', 'extra_people']
    for col in to_change_to_0:
        df[col].fillna(value=0,inplace=True)

    v = [list(i) for i in df.as_matrix()]
    return v

def separate_cols(files):
    '''
        Separates out the different column types into 3 lists of column names.

        input: list of airbnb data files
        output: a tuple of length three
    '''

    cols = get_col_names(files)

    label_col = ['price']
    # ones that are never null
    categorical_cols = ['require_guest_profile_picture', 'require_guest_phone_verification', 'requires_license', 'instant_bookable', 'bed_type', 'cancellation_policy', 'room_type']
    num_cols = ['number_of_reviews', 'accommodates', 'minimum_nights', 'maximum_nights', 'guests_included', 'availability_30', 'availability_60', 'availability_90', 'availability_365', 'reviews_per_month', 'beds', 'bedrooms', 'bathrooms', 'host_response_rate', 'host_acceptance_rate',  'review_scores_accuracy', 'review_scores_communication', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_rating', 'review_scores_value', 'review_scores_checkin',                'security_deposit', 'cleaning_fee', 'extra_people']

    # c nulls to ""
    text_cols = ['name', 'neighborhood_overview', 'summary', 'transit', 'street', 'host_neighbourhood', 'notes', 'space', 'description']

    return label_col+categorical_cols, label_col+num_cols, label_col+text_cols

def save_sparse_csr(filename,array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
 shape = loader['shape'])

def save_datasets(col_type, vector):
    '''
        input:
            - col_type: categorical, text, num
            - featurized vector (not split into text, train)

        Saves 1 file with col_type prefix
    '''
    if col_type == 'text':
        save_sparse_csr('data/' + col_type + '_features.sparse', vector)
    else:
        with open('data/' + col_type + '_features.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(vector)

def create_datasets(max_price, text_type_vec='count', text_max_feats=250):
    '''
        Creates train, dev, and test files for each column type (categorical, num, text)
    '''

    # this could be cleaned up with a loop for modularity

    categorical_cols, num_cols, text_cols = separate_cols(airbnb_files)
    all_text_df = pd.DataFrame()
    all_num_df = pd.DataFrame()
    all_categorical_df = pd.DataFrame()

    for f in airbnb_files:
        df = pd.read_csv(f, dtype={'zipcode': 'str'})

        categorical_df = df[categorical_cols]
        num_df = df[num_cols]
        text_df = df[text_cols]

        all_text_df = all_text_df.append(text_df)
        all_num_df = all_num_df.append(num_df)
        all_categorical_df = all_categorical_df.append(categorical_df)

    text_vector = featurize_text(all_text_df, max_price, text_type_vec, text_max_feats)
    categorical_vector = featurize_categorical(all_categorical_df, max_price)
    num_vector = featurize_num(all_num_df, max_price)

    save_datasets('categorical', categorical_vector)
    save_datasets('num', num_vector)
    save_datasets('text', text_vector.tocsr())

def append_column_wise(orig, to_append):
    for item,lst in zip(to_append,orig):
            lst.insert(0,item)
            print lst
    return orig

def read_files_to_datasets(files):
    '''
        Returns X and Y
    '''

    X = None
    y = None
    y_added = False

    for f in files:
        if '.csv' in f:
            to_add = np.loadtxt(open(f, "rb"), delimiter=",")
        else:
            to_add = load_sparse_csr(f)
        if y_added:
            to_add = np.delete(to_add, 0, 1)
            X = np.hstack((X, to_add))
        else:
            if type(to_add) is np.ndarray:
                y = to_add[:, [0]]
                y = np.asarray(y).reshape(-1)
                to_add = np.delete(to_add, 0, 1)
            else:
                y = to_add[:, [0]].todense()
                y = np.asarray(y).reshape(-1)
                to_add = np.delete(to_add.toarray(), 0, 1)

            X = to_add
            y_added = True
    return X, y

def lin_reg(X_train, Y_train, X_test, Y_test, max_price, plot=False):
    # Create linear regression object
    regr = LinearRegression()

    # Train the model using the training sets
    regr.fit(X_train, Y_train)

    print('Coefficients: \n', regr.coef_) # coefficients of everything but text

    predictions = regr.predict(X_test).astype(np.float64)
    predictions = np.maximum(np.minimum(predictions, max_price), 0.) # make sure prices don't go out of range

    if plot:
        plt.scatter(Y_test, predictions)
        plt.xlim(0,max_price)
        plt.ylim(0,max_price)
        plt.xlabel("Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Prices vs. Predicted Prices")
        plt.show()

    print "predictions min: " + str(np.amin(predictions))
    print "predictions max: " + str(np.amax(predictions))

    msq = mean_squared_error(Y_test, predictions)
    print("Mean squared error: %.2f" % msq)
    # Explained variance score: 1 is perfect prediction
    print("Root mean squared error: %.2f" % math.sqrt(msq))
    print('Variance score: %.2f' % regr.score(X_test, Y_test))

def run_model(max_price=500, text_type_vec='count', text_max_feats=500, plot=False, train_size=0.85):
    create_datasets(max_price, text_type_vec=text_type_vec, text_max_feats=text_max_feats)
    X, y = read_files_to_datasets(['data/text_features.sparse.npz','data/num_features.csv', 'data/categorical_features.csv'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)

    print "*****************************************************"
    print "MAX PRICE: " + str(max_price)
    print "TEXT MAX FEATURES: " + str(text_max_feats)
    print "TEXT VECTORIZER TYPE: " + text_type_vec
    print "-----------------------------------------------------"

    print "y_test std dev: " + str(math.sqrt(np.var(y_test)))
    print "y_train std dev: " + str(math.sqrt(np.var(y_train)))

    print "y_test mean: " + str(np.mean(y_test))
    print "y_train mean: " + str(np.mean(y_train))

    print "number of training datapoints: " + str(X_train.shape[0])
    print "number of features: " + str(X_train.shape[1])

    lin_reg(X_train, y_train, X_test, y_test, max_price, plot=plot)

    print "*****************************************************"

def grid_search():
    TRAIN_SIZE = 0.85
    CROSS_VALIDATION = False
    CROSS_VALIDATION_COUNT = 2
    max_prices = [150.0, 250.0, 350.0, 500.0, 750.0, 1000.0]
    possible_text_vects = ['count', 'tfidf']
    possible_max_features = [250, 300, 400, 500, 600, 750]

    f = open("ML_output.txt", "a")

    for price in max_prices:
        for vec_type in possible_text_vects:
            for max_features in possible_max_features:

                print "*****************************************************"
                print "MAX PRICE: " + str(price)
                print "TEXT MAX FEATURES: " + str(max_features)
                print "TEXT VECTORIZER TYPE: " + vec_type
                print "-----------------------------------------------------"

                create_datasets(price, text_type_vec=vec_type, text_max_feats=max_features)
                X, y = read_files_to_datasets(['data/text_features.sparse.npz','data/num_features.csv', 'data/categorical_features.csv'])

                # cross validation loop
                if CROSS_VALIDATION:
                    for i in range(CROSS_VALIDATION_COUNT):
                        print "-------- split #" + str(i + 1) + "--------\b"
                        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)

                        print "y_test std dev: " + str(math.sqrt(np.var(y_test)))
                        print "y_train std dev: " + str(math.sqrt(np.var(y_train)))

                        print "y_test mean: " + str(np.mean(y_test))
                        print "y_train mean: " + str(np.mean(y_train))

                        print "number of training datapoints: " + str(X_train.shape[0])
                        print "number of features: " + str(X_train.shape[1])

                        lin_reg(X_train, y_train, X_test, y_test, price)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE)

                    print "y_test std dev: " + str(math.sqrt(np.var(y_test)))
                    print "y_train std dev: " + str(math.sqrt(np.var(y_train)))

                    print "y_test mean: " + str(np.mean(y_test))
                    print "y_train mean: " + str(np.mean(y_train))

                    print "number of training datapoints: " + str(X_train.shape[0])
                    print "number of features: " + str(X_train.shape[1])

                    lin_reg(X_train, y_train, X_test, y_test, price)
                    print "*****************************************************"

if __name__ == '__main__':
    run_model()

