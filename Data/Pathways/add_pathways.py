import pandas as pd
import requests

pathway_ids = ['hsa05221', 'hsa05219', 'hsa05224', 'hsa05210', 'hsa05212', 'hsa05225',
               'hsa05226', 'hsa05214', 'hsa05216', 'hsa05220', 'hsa05217', 'hsa05218',
               'hsa05211', 'hsa05215', 'hsa05213', 'hsa05222', 'hsa05223', 'hsa05206',
               'hsa05200']

df = pd.DataFrame(columns=['pathway', 'gene'])

for pathway in pathway_ids:
    url = f"https://rest.kegg.jp/get/pathway:{pathway}"
    r = requests.get(url)
    r = r.content.decode('utf-8')

    print(f"PATHWAY: {pathway}")
    print(f"{pathway_ids.index(pathway)} / {len(pathway_ids)} PATHWAYS")
    
    # Get the networks
    networks = []
    in_network = False
    in_element = False
    for line in r.split('\n'):
        if line.split()[0] == 'NETWORK':
            in_network = True
            continue
        if in_network and line.split()[0] == 'ELEMENT':
            networks.append(line.split()[1])
            in_element = True
            continue
        if line.split()[0] == 'DISEASE':
            in_network = False
            in_element = False
            break
        if in_element and in_network:
            networks.append(line.split()[0])

    for network in networks:
        print(f"NETWORK: {network}")
        print(f"{networks.index(network)} / {len(networks)} NETWORKS")
        url = f'https://rest.kegg.jp/get/network:{network}'
        r2 = requests.get(url)
        r2 = r2.content.decode('utf-8')

        genes = []
        in_gene = False
        for line in r2.split('\n'):
            if line.split()[0] == 'GENE':
                genes.append(line.split()[1])
                in_gene = True
                continue
            if line.split()[0] == 'VARIANT':
                in_gene = False
                break
            if in_gene:
                genes.append(line.split()[0])

        ids = []
        for gene in genes:
            print(f"Gene: {gene}")
            print(f"{genes.index(gene)} / {len(genes)} GENES")
            url = f'https://rest.kegg.jp/get/hsa:{gene}'
            r3 = requests.get(url)
            r3 = r3.content.decode('utf-8')

            in_dblinks = False
            for line in r3.split('\n'):
                if line.split()[0] == 'DBLINKS':
                    in_dblinks = True
                    continue
                if in_dblinks and line.split()[0] == 'Ensembl:':
                    ids.append(line.split()[1])
                    in_dblinks = False
                    break
                # if line.split()[0] doesn't end with a colon
                if in_dblinks and line.split()[0][-1] != ':':
                    in_dblinks = False
                    break

        print(ids)
        # Append to a dataframe
        for id in ids:
            # If the network and gene are already in the dataframe, skip
            if df[(df['pathway'] == network) & (df['gene'] == id)].shape[0] > 0:
                continue
            df2 = pd.DataFrame({'pathway': [network], 'gene': [id]})
            df = pd.concat([df, df2], ignore_index=True)

print("Dataframe shape:", df.shape)

df.to_csv("pathway_genes.csv", index=False)




