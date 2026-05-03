import pandas as pd
import math
import numpy as np
import json

from langdetect import detect
import stanza
stanza.download('id')

stanza_nlp = stanza.Pipeline('id', use_gpu=False)

df = pd.read_csv("../preprocessed/dataset_preprocessed.csv")

df["tokens"] = [[] for _ in range(len(df))]

CHUNK_SIZE = 750

for index, row in df.iterrows():
    while True:
        try:
            print("index:", index)
            
            text = row["konten_clean"]

            # split text into chunks
            chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

            for chunk in chunks:
                print(len(chunk))
                doc = stanza_nlp(chunk)
                
                for sentence in doc.sentences:
                    for word in sentence.words:
                        df.loc[index, "tokens"].append((word.text, word.upos))
            
            break

        except Exception as e:
            print(e)

# create a copy and save the POS as json
df_output = df.copy()
df_output["tokens_json"] = df_output["tokens"].apply(json.dumps)
df_output = df_output.drop(columns=["tokens"])
df_output.to_csv("../outputs/part_of_speech.csv", index=False)