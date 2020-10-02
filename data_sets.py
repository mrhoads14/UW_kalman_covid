import pandas as pd


def create_case_df_county():
    data = pd.read_csv(r'data/us-counties.csv', dtype={'fips': 'str',
                                                       'cases': 'int64',
                                                       'deaths': 'int64'})
    data['date'] = pd.to_datetime(data['date'], yearfirst=True)
    date_rng = pd.date_range(data['date'].min(), data['date'].max())

    # we do not need rows where the county is unknown (we have a diff dataset
    # for state values)
    data.drop(data.index[data['county'] == 'Unknown'], axis=0, inplace=True)
    # to ensure the fips value is unique across the dataframe, use the county
    # name as the fips value (this is for special cases like NYC)
    blank_fips = data['fips'].isnull()
    data.loc[blank_fips, 'fips'] = data.loc[blank_fips, 'county']

    # creates the multiindex using fips and date (now that these values uniquely
    # identify each row)
    data.set_index(['fips', 'date'], inplace=True)
    data.sort_index(inplace=True)

    # standardizes the date range for each fips
    iterables = [data.index.levels[0], date_rng]
    new_index = pd.MultiIndex.from_product(iterables, names=['fips', 'date'])
    data = data.reindex(new_index)

    # fixes data values now that it has been reindexed
    data['county'].fillna(method='bfill', inplace=True)
    data['state'].fillna(method='bfill', inplace=True)
    data['cases'].fillna(value=0, inplace=True)
    data['deaths'].fillna(value=0, inplace=True)

    # creates the case rate column
    data['case_rate'] = data['cases'].groupby(level=0).diff()
    # sets initial case_rate values to the number of cases
    nan_vals = data['case_rate'].isnull()
    data.loc[nan_vals, 'case_rate'] = data.loc[nan_vals, 'cases']
    data[['cases', 'deaths', 'case_rate']] = data[['cases',
                                                   'deaths',
                                                   'case_rate']].astype(
                                                       'int64')
    return data


def create_case_df_state():
    data = pd.read_csv(r'data/us-states.csv', dtype={'fips': 'str',
                                                     'cases': 'int64',
                                                     'deaths': 'int64'})
    data['date'] = pd.to_datetime(data['date'], yearfirst=True)
    date_rng = pd.date_range(data['date'].min(), data['date'].max())

    # sets the multiindex
    data.set_index(['state', 'date'], inplace=True)
    data.sort_index(inplace=True)
    # standardizes the date range for each fips
    iterables = [data.index.levels[0], date_rng]
    new_index = pd.MultiIndex.from_product(iterables, names=['state', 'date'])
    data = data.reindex(new_index)

    # fills in the missing values created by the reindexing
    data['fips'].fillna(method='bfill', inplace=True)
    data['cases'].fillna(value=0, inplace=True)
    data['deaths'].fillna(value=0, inplace=True)

    # creates the case_rate column
    data['case_rate'] = data['cases'].groupby(level=0).diff()
    nan_vals = data['case_rate'].isnull()
    data.loc[nan_vals, 'case_rate'] = data.loc[nan_vals, 'cases']
    data[['cases', 'deaths', 'case_rate']] = data[['cases',
                                                   'deaths',
                                                   'case_rate']].astype(
                                                       'int64')
    return data


def create_symptom_df():
    data = pd.read_csv(r'data/fb/fb_smell.csv',
                       dtype={'fips': 'str',
                              'n': 'int64',
                              'smell_taste_loss': 'float64'})
    data.drop(data.columns[0], axis=1, inplace=True)
    data['fips'] = data['fips'].str.replace('(^[0-9]{4}\.0)', '0\g<1>')
    data['fips'] = data['fips'].str.extract('(^[0-9]{5})')

    data['2l'] = data['fips'].str.slice(start=0, stop=2)
    data['state'] = data['2l'].map(FIPS_TO_STATE)
    data.drop(['2l'], axis=1, inplace=True)

    data['date'] = pd.to_datetime(data['date'], yearfirst=True)
    date_rng = pd.date_range(data['date'].min(), data['date'].max())
    # creates the multiindex using fips and date (now that these values uniquely
    # identify each row)
    data.set_index(['fips', 'date'], inplace=True)
    data.sort_index(inplace=True)

    # standardizes the date range for each fips
    iterables = [data.index.levels[0], date_rng]
    new_index = pd.MultiIndex.from_product(iterables, names=['fips', 'date'])
    data = data.reindex(new_index)
    data['n'].fillna(value=0, inplace=True)
    data['smell_taste_loss'].fillna(value=0, inplace=True)
    data['state'].fillna(method='bfill', inplace=True)

    # adds state to the multiindex and moves it to the highest index level
    data.set_index('state', append=True, inplace=True)
    data = data.reorder_levels(order=['state', 'fips', 'date'])
    data.sort_index(inplace=True)

    data['num_stl'] = data['n'].mul(data['smell_taste_loss'])
    data['num_stl'] = data['num_stl'].round()
    data['num_stl'] = data['num_stl'].astype('int64')
    data['n'] = data['n'].astype('int64')
    # get rid of the proportion column because further transformations will make
    # it inaccurate -- should be recreated when needed
    data.drop('smell_taste_loss', axis=1, inplace=True)

    return data


def get_fips(state_2l, name):
    """
    Given a two letter designator and the name of a county, returns the
    appropriate 5 digit FIPS code (in the form of a string).
    """
    data = pd.read_csv(r'census_bureau/2019_Gaz_counties_national.txt',
                       sep='\t', encoding='latin_1', dtype={'GEOID': 'str'})
    data.rename(columns={data.columns[-1]: 'INTPTLONG'}, inplace=True)

    data['NAME'] = data['NAME'].str.replace(' County', '')
    data['NAME'] = data['NAME'].str.replace(' Parish', '')
    data['NAME'] = data['NAME'].str.replace(' Municipality', '')

    state = data[data['USPS'] == state_2l]
    fips = state[state['NAME'] == name]['GEOID'].iloc[0]
    return fips


def get_pop_county(fips):
    if fips == '36001':
        return 305506
    else:
        return None


def get_pop_state(state):
    if state == 'New York':
        return 19453561
    else:
        return None


STATES = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DE': 'Delaware',
    'DC': 'District of Columbia',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
    'VI': 'Virgin Islands',
    'GU': 'Guam',
    'MP': 'Northern Mariana Islands'
    }


STATE_FIPS = {
    'AL': '01',
    'AK': '02',
    'AZ': '04',
    'AR': '05',
    'CA': '06',
    'CO': '08',
    'CT': '09',
    'DE': '10',
    'DC': '11',
    'FL': '12',
    'GA': '13',
    'HI': '15',
    'ID': '16',
    'IL': '17',
    'IN': '18',
    'IA': '19',
    'KS': '20',
    'KY': '21',
    'LA': '22',
    'ME': '23',
    'MD': '24',
    'MA': '25',
    'MI': '26',
    'MN': '27',
    'MS': '28',
    'MO': '29',
    'MT': '30',
    'NE': '31',
    'NV': '32',
    'NH': '33',
    'NJ': '34',
    'NM': '35',
    'NY': '36',
    'NC': '37',
    'ND': '38',
    'OH': '39',
    'OK': '40',
    'OR': '41',
    'PA': '42',
    'PR': '72',
    'RI': '44',
    'SC': '45',
    'SD': '46',
    'TN': '47',
    'TX': '48',
    'UT': '49',
    'VT': '50',
    'VA': '51',
    'WA': '53',
    'WV': '54',
    'WI': '55',
    'WY': '56',
    'VI': '78',  # Virgin Islands
    'GU': '66',  # Guam
    'MP': '69'  # Northern Mariana Islands
    }


FIPS_TO_STATE = {
    '01': STATES['AL'],
    '02': STATES['AK'],
    '04': STATES['AZ'],
    '05': STATES['AR'],
    '06': STATES['CA'],
    '08': STATES['CO'],
    '09': STATES['CT'],
    '10': STATES['DE'],
    '11': STATES['DC'],
    '12': STATES['FL'],
    '13': STATES['GA'],
    '15': STATES['HI'],
    '16': STATES['ID'],
    '17': STATES['IL'],
    '18': STATES['IN'],
    '19': STATES['IA'],
    '20': STATES['KS'],
    '21': STATES['KY'],
    '22': STATES['LA'],
    '23': STATES['ME'],
    '24': STATES['MD'],
    '25': STATES['MA'],
    '26': STATES['MI'],
    '27': STATES['MN'],
    '28': STATES['MS'],
    '29': STATES['MO'],
    '30': STATES['MT'],
    '31': STATES['NE'],
    '32': STATES['NV'],
    '33': STATES['NH'],
    '34': STATES['NJ'],
    '35': STATES['NM'],
    '36': STATES['NY'],
    '37': STATES['NC'],
    '38': STATES['ND'],
    '39': STATES['OH'],
    '40': STATES['OK'],
    '41': STATES['OR'],
    '42': STATES['PA'],
    '72': STATES['PR'], # Puerto Rico
    '44': STATES['RI'],
    '45': STATES['SC'],
    '46': STATES['SD'],
    '47': STATES['TN'],
    '48': STATES['TX'],
    '49': STATES['UT'],
    '50': STATES['VT'],
    '51': STATES['VA'],
    '53': STATES['WA'],
    '54': STATES['WV'],
    '55': STATES['WI'],
    '56': STATES['WY'],
    '78': STATES['VI'],  # Virgin Islands
    '66': STATES['GU'],  # Guam
    '69': STATES['MP']  # Northern Mariana Islands
    }
