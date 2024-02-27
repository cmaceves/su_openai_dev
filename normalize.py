
import time
from typing import Sequence

import requests

#node_normalizer_url = 'https://nodenormalization-sri.renci.org/get_normalized_nodes'
node_normalizer_url = "https://nodenorm.transltr.io/get_normalized_nodes"
#curies = ['P01286']
#response = requests.post(node_normalizer_url, json={"curies": curies})
#r = response.json()
#for x in r[curies[0]]['equivalent_identifiers']:
#    print(x['identifier'])
def get_normalizer(curies: Sequence[str]):
    """
    Given a list of CURIEs `curies`, build and return a `normalizer` dictionary,
    which maps all the CURIEs equivalent to those in `curies` each to its
    equivalent preferred CURIE specified by the Node Normalizer.

    Args:
        curies: (sequence of str) CURIEs that we need to find all equivalent
            CURIEs of
    
    Returns:
        dict that maps CURIE -> equivalent preferred CURIE
    """
    while True:
        response = None
        try:
            response = requests.post(node_normalizer_url, json={"curies": curies})
            result = response.json()
        except:
            if response is not None:
                print(f'{response.status_code} {response.reason}')
            print('Request to the Node Normalizer failed. Retrying in 5 seconds...')
            time.sleep(5)
        break
    normalizer = {}
    all_norm = []
    for curie, entry in result.items():
        if entry is None:
            continue
        for equiv in entry['equivalent_identifiers']:
            normalizer[equiv['identifier']] = curie
            if 'label' in equiv:
                all_norm.append(equiv['label'])
    return (all_norm, normalizer)

if __name__ == '__main__':
    get_normalizer(['MESH:D012223', 'MESH:D003233'])
