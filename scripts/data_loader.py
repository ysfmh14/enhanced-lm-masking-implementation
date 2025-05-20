import pandas as pd

def load_data(data_path: str):
    df = pd.read_excel(data_path)
    return df

def load_lexicon(path: str):
    df_entitie = pd.read_excel(path)
    df_entitie = df_entitie.drop_duplicates(subset=['Name'], keep='first')
    df_entitie["Name"] = df_entitie["Name"].astype(str).str.lower()
    return df_entitie