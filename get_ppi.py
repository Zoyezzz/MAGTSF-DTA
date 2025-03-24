import requests
import json
import pickle


uniprot_ids = ['P12345', 'Q67890', 'O98765']  


def get_string_interactions(uniprot_ids, confidence=700):  
    string_api_url = "https://string-db.org/api/json/interactions"
    params = {
        "identifiers": "%0d".join(uniprot_ids),
        "species": 9606, 
        "limit": 100,
        "required_score": confidence,
    }
    response = requests.post(string_api_url, data=params)

    if response.status_code == 200:
        interactions = json.loads(response.text)
        return interactions
    else:
        print(f"Error fetching data from STRING: {response.status_code}")
        return []

string_interactions = get_string_interactions(uniprot_ids)

def get_intact_interactions(uniprot_ids):
    intact_api_url = "https://www.ebi.ac.uk/intact/ws/rest/interactions/findProteins/"  
    proteins_query = ','.join(uniprot_ids) 
    response = requests.get(intact_api_url + proteins_query + '?complexExpansion=true&returnType=json')  

    if response.status_code == 200:
        intact_data = json.loads(response.text)
        return intact_data
    else:
        print(f"Error fetching data from IntAct: {response.status_code}")
        return []

intact_interactions = get_intact_interactions(uniprot_ids)


def get_interpro_features(uniprot_id):

    return {"feature1": "value1", "feature2": "value2"}  

interpro_features = {uid: get_interpro_features(uid) for uid in uniprot_ids}

import numpy as np

n = len(uniprot_ids)
adjacency_matrix = np.zeros((n, n))
id_to_index = {uid: i for i, uid in enumerate(uniprot_ids)}


for interaction in string_interactions:
    protein1 = interaction['preferredName_A'] 
    protein2 = interaction['preferredName_B']
    if protein1 in id_to_index and protein2 in id_to_index:
        i = id_to_index[protein1]
        j = id_to_index[protein2]

        combined_score = interaction['combined_score'] 
        adjacency_matrix[i, j] = combined_score / 1000  
        adjacency_matrix[j, i] = combined_score / 1000  


for interaction in intact_interactions:

    proteins = interaction.get('proteins', [])
    if len(proteins) == 2:
      protein1 = proteins[0].get('identifier')
      protein2 = proteins[1].get('identifier')

      if protein1 in id_to_index and protein2 in id_to_index:
        i = id_to_index[protein1]
        j = id_to_index[protein2]

        adjacency_matrix[i, j] = 1.0  
        adjacency_matrix[j, i] = 1.0


data = {
    'adjacency_matrix': adjacency_matrix,
    'interpro_features': interpro_features,
    'string_interactions': string_interactions,
    'intact_interactions': intact_interactions
}

with open('ppi_network_data.pkl', 'wb') as f:
    pickle.dump(data, f)

print("PPI network data saved to ppi_network_data.pkl")