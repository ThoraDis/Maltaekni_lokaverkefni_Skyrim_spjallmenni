import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer

BUTAR_SKJAL="butar.json"
GREYP_SKJAL="greypingar.index"
GREYP_MODEL='sentence-transformers/all-MiniLM-L6-v2'

def leita(model,index,spurning):
    spurning_greyping=model.encode(spurning)
    vector=np.array([spurning_greyping], dtype="float32")
    faiss.normalize_L2(vector) 

    D, I = index.search(vector, 3)

    return D, I


#Fyrir mannleg svör: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

def qwen_svar(butar,spurning,model,tokenizer): 
    messages = [ {"role": "user", "content": f""" 
                  Answer the question using the context below. 
                  
                  Respond like a human would 
                  Combine information from each context 
                  Do not list chunks separately 
                  Remove repetition 
                  Question: {spurning} Context 1: {butar[0]} Context 2: {butar[1]} Context 3: {butar[2]} """},
    ]
    inputs = tokenizer.apply_chat_template(
	    messages,
	    add_generation_prompt=True,
	    tokenize=True,
	    return_dict=True,
	    return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=160)
    return tokenizer.decode( outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True)


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
qwen_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )


#Ná í greypingar og buta
index=faiss.read_index(GREYP_SKJAL)

with open(BUTAR_SKJAL, 'r') as F:
    butar = json.load(F)

print("I am a chatbot that answers questions about the game Skyrim, what would you like to know:")
#Fá input
spurning=input()

#Leita í faiss

greypa_model = SentenceTransformer(GREYP_MODEL)

D,I=leita(greypa_model,index,spurning)

butar=[butar[idx] for idx in I[0]]

svar=qwen_svar(butar,spurning,qwen_model,tokenizer)
print(f"Answer: {svar}")
