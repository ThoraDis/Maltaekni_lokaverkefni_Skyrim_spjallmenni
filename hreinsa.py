import re
import os

def hreinsa_markdown(text):

    text = re.sub(r"^#\s*", "## ", text)
    text = re.sub(r"####\s*", " ", text)        
    text = re.sub(r"###\s*", "## ", text)       
    text = re.sub(r"---", "", text)
    text = re.sub(r"_(.*?)_", r"\1", text)

    text = laga_toflur(text)

    text = text.strip()

    return text

def laga_toflur(texti):
    linur=texti.split("\n")
    lagfaera=[]

    for lina in linur:
        if "|" in lina:

            hlutir=lina.split("|")

            if len(hlutir)==2:
                lina=f"{hlutir[0]}: {hlutir[1]}"

            elif len(hlutir)==4:
                lina=f"{hlutir[0]}: {hlutir[1]} | {hlutir[2]}: {hlutir[3]} "
    
        lagfaera.append(lina)

    return "\n".join(lagfaera)


def lesa_moppu(markdown_mappa, rag_mappa):
    os.makedirs(rag_mappa)

    for filename in os.listdir(markdown_mappa):

        markdown_path = os.path.join(markdown_mappa, filename)
        rag_path = os.path.join(rag_mappa, filename)

        with open(markdown_path, "r", encoding="utf-8") as f:
            texti = f.read()

        hreinsud_skra = hreinsa_markdown(texti)

        with open(rag_path, "w", encoding="utf-8") as f:
            f.write(hreinsud_skra)


lesa_moppu("UESP_Wiki/markdown/Skyrim_md", "RAG_data")