import pandas as pd
import requests

def get_ticker_cik_map():
    """Retrieves the SEC's ticker-CIK mapping file as a dataframe."""
    df = pd.read_csv('https://www.sec.gov/include/ticker.txt', sep='\t', names=['ticker', 'cik'])
    df['cik'] = df['cik'].astype(str).apply(pad_cik)
    return df

def pad_cik(cik):
    """"Pads the beginning of a CIK with zeros if it has fewer than 10 digits."""
    digits_missing = 10 - len(cik)
    if digits_missing == 0:
        return cik
    else:
        zeros = '0' * digits_missing
        return zeros + cik

def get_data(url, headers):
    """GET requests a URL and returns the JSON results as a dictionary."""
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()

def get_company_facts(headers, cik):
    """Calls the SEC's company facts API to get data on a company's facts."""
    url = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{0}.json'.format(cik)
    facts = get_data(url, headers=headers)
    if facts is not None:
        return facts

def get_company_schema(company_facts):
    """Extracts the schema from a company's facts data."""
    if company_facts is not None:
        schema = {
            'taxonomy': [],
            'tag': [],
            'unit': [],
            'description': []
        }
        if 'facts' in company_facts:
          for taxonomy, facts in company_facts['facts'].items():
              for tag, attributes in facts.items():
                  description = attributes['description']
                  for unit in attributes['units'].keys():
                      schema['taxonomy'].append(taxonomy)
                      schema['tag'].append(tag)
                      schema['unit'].append(unit)
                      schema['description'].append(description)
          return pd.DataFrame(schema)

def sample_company_schemas(headers, ticker_cik_map, n=10, random_state=None):
    """Randomly samples n companies' schemas and counts how many include each concept."""
    sample = ticker_cik_map['cik'].sample(n, random_state=random_state)
    schemas = []
    for cik in sample:
        facts = get_company_facts(headers, cik)
        schema = get_company_schema(facts)
        schemas.append(schema)
    sample_schema = pd.concat(schemas)
    sample_schema = (sample_schema
                      .groupby(['taxonomy', 'tag', 'unit', 'description']) # group by concept 
                      .size() # count the number of companies that report the concept 
                      .reset_index()
                      .rename(columns={0: 'companies_reported'})
                      .sort_values('companies_reported', ascending=False) # sort the dataframe by the most commonly reported concepts
                      .reset_index(drop=True)
                    )
    return sample_schema

def get_concept(headers, taxonomy, tag, unit, period):
    """Calls the SEC's frames API to get data on the given concept (taxonomy-tag-unit combination) and period."""
    url = 'https://data.sec.gov/api/xbrl/frames/{0}/{1}/{2}/{3}.json'.format(taxonomy, tag, unit, period)
    response = get_data(url, headers=headers)
    if response is not None:
        df = pd.DataFrame(response['data'])
        df['taxonomy'] = taxonomy
        df['tag'] = tag 
        df['unit'] = unit
        df['period'] = period
        return df

def get_all_concepts(period, schema):
    """Calls the SEC's frames API to get data on the given period and concepts listed in the provided schema."""
    frames = []
    for i, row in schema.iterrows(): 
        frame = get_concept(headers, row['taxonomy'], row['tag'], row['unit'], period)
        frames.append(frame)
    df_long = pd.concat(frames)
    df_long['concept'] = df_long.apply(lambda row: row['tag'] + '_' + row['unit'], axis=1)
    # pivot the dataframe from long form (each row is a company-concept combination) to wide form (each row is a company and each column is a concept)
    df_wide = df_long.pivot(
        index='cik',
        columns='concept',
        values='val'
    )
    return df_wide
