import os
import re
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

BUTAR_SKJAL="butar.json"
GREYP_SKJAL="greypingar.index"
GREYP_MODEL='sentence-transformers/all-MiniLM-L6-v2'

#Ná í md skjöl

def lesa_markdown_moppu(mappa):
    count=0
    butar=[]

    skjol=os.listdir(mappa)

    for skjal in skjol:
        with open(os.path.join(mappa,skjal),"r", encoding='utf-8') as f:
            texti=f.read()
            chunk=markdown_skjal_i_buta(texti,skjal)
            butar.extend(chunk)

            count +=1
            if count%100==0:
                print(count/len(skjol))

    return butar


#Taka section til að gera meiri specific, búta

def markdown_skjal_i_buta(texti,skjal):
    buta_skipti=re.split(r"\n##\s+",texti)
    butar=[]

    for i in range(0,len(buta_skipti)):

        hluti=buta_skipti[i].split("\n",1)

        heiti=hluti[0].strip().lower().replace(" ","_")

        if heiti == "notes" or heiti =="bugs":
            continue

        innihald=hluti[1].strip() if len(hluti) > 1 else ""

        butar.append(
            f"name:{skjal} - {heiti}: {innihald}"
        )
    
    return butar

#Gera greypingar - https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2  byggt á bert

def greypa(model,butar):

    greypingar = model.encode(
        butar,
        batch_size=256,
        show_progress_bar=True
    )
    return greypingar


#Geyma með FAISS https://dev.to/the_aayush_mishra/building-a-local-documentation-chatbot-with-python-faiss-and-openai-55oc  https://gabrieltardochi.medium.com/nlp-efficient-semantic-similarity-search-with-faiss-facebook-ai-similarity-search-and-gpus-274771d0709a
def FAISS_geymsla(greypingar):
    vector=np.array(greypingar, dtype="float32")
    faiss.normalize_L2(vector)
    index = faiss.IndexFlatIP(vector.shape[1])
    index.add(vector)

    faiss.write_index(index, GREYP_SKJAL)

    return index 


#Ná í búta og geyma
print("Búta texta")
butar=lesa_markdown_moppu("RAG_data")

print(len(butar))
print("Geyma búta")
with open(BUTAR_SKJAL, 'w') as f:
    json.dump(butar, f)

#Gera greypingar og geyma
print("Gera greypingar")
model = SentenceTransformer(GREYP_MODEL)
greypingar=greypa(model,butar)
FAISS_geymsla(greypingar)