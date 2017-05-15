# Notes

## Dataset
Datapoints from each city are randomly placed into training and test

8 US Cities:

  - Chicago, IL
  - New Orleans, LA
  - Los Angeles, CA
  - Portland, OR
  - Nashville, TN
  - Asheville, NC
  - Austin, TX
  - Boston, MA

## Development Dataset

## Testing Dataset

## Housing Cost Dataset
Used to account for differences in cost of livings in different zip codes

Source: U.S. Census Bureau, 2011-2015 American Community Survey 5-Year Estimates
Universe: Owner-occupied housing units

## Notes on columns:

	Looks like the austin data has 15 rows with deleted (?) hosts

    CATEGORICAL VALS:
        - require_guest_profile_picture --> [t, f]
        - require_guest_phone_verification --> [t, f]
        - requires_license --> [t, f]
        - instant_bookable --> [t, f]
        - bed_type --> ['Real Bed' 'Futon' 'Airbed' 'Pull-out Sofa' 'Couch']
        - cancellation_policy --> ['moderate' 'flexible' 'strict' 'super_strict_30' 'no_refunds']
        - room_type --> ['Private room' 'Entire home/apt' 'Shared room']

        - NEED TO COERCE:
            * host_identity_verified --> [t, f, nan] (change nan to f -- or get rid of aus 15)
            * host_is_superhost --> [t, f, nan] (change nan to f -- or get rid of aus 15)
            * host_has_profile_pic --> [t, f, nan] (change nan to f -- or get rid of aus 15)
            * property_type --> limited number of options, figure out the union of them
            * amenities --> limited number of options, figure out the union of them
            * host_verifications --> limited number of options, but the data is in a list stored as a string?
            * host_response_time --> [nan 'within a few hours' 'within an hour' 'within a day' 'a few days or more']

    NUM VALS:
        ** change all num values to floats

        - number_of_reviews --> [0 or greater]
        - accommodates --> [0, 16]
        - minimum_nights --> [1 on]
        - maximum_nights --> [1 on] (some arbitrary vals, though... i.e. 9999999 vs. like 26000 mean kind of the same thing. Threshold it?)
        - guests_included --> [0 on]
        - availability_30 --> [0 on]
        - availability_60 --> [0 on]
        - availability_90 --> [0 on]
        - availability_365 --> [0 on]

        - NEED TO COERCE:
            * beds: [0 or greater, nan] (change nan to 0? does nan mean not reported?)
            * host_total_listings_count: [0 or greater, nan] (change nan to 1 because each host must have 1 listing?)
            * host_response_rate: [0-100%, nan] (change nan to?)
            * host_acceptance_rate: [0-100%, nan] (change nan to?) -- haven't gotten any requests?
            * reviews_per_month --> float value, change NaNs to 0
            * bathrooms --> [0 on, nans] (guess using nearest neighbors? use 0?)
            * review_scores_accuracy --> 0 to 10 or NaN
            * review_scores_communication --> 0 to 10 or NaN
            * review_scores_cleanliness --> 0 to 10 or NaN
            * review_scores_location --> 0 to 10 or NaN
            * review_scores_rating --> 0 to 10 or NaN
            * review_scores_value --> 0 to 10 or NaN
            * review_scores_checkin --> 0 to 10 or NaN
            * security_deposit --> dollar values, strip $ and ,'s. Change nan's to 0
            * extra_people --> dollar values, never null (need to clean and change to nums)
            * cleaning_fee --> dollar values, strip $ and ,'s. Change nan's to 0
            * bedrooms --> change nans to 0 or 1... usually studios?

    TEXT VALS: (coerce all nans to "")
        - name --> always a string, never null
        - street --> always a string, never null (includes street, city, state, zip, country)

        - NEED TO COERCE:
            * transit --> string, nan's to ""
            * summary --> string, nan's to ""
            * neighborhood_overview --> string, nan's to ""
            * host_neighbourhood
            * notes
            * space
            * description
