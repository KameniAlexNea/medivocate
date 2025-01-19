from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama

TEMPLATE = """
J'ai un prompt posé par un utilisateur destiné à récupérer des informations à partir d'un système de génération augmentée par la récupération (RAG), où des segments de documents sont stockés sous forme d'embeddings pour une recherche efficace et précise. Votre tâche consiste à affiner ce prompt afin de :

1. Améliorer la pertinence de la recherche en alignant la requête avec la granularité sémantique et l'intention des embeddings.
2. Minimiser l'ambiguïté pour réduire le risque de récupérer des segments non pertinents ou trop génériques.
3. Préserver autant que possible le langage, le ton et la structure du prompt original tout en le rendant plus clair et efficace.

Voici le prompt original de l'utilisateur :
{user_prompt}  

Instructions :

- Réécrivez le prompt pour améliorer sa clarté et son alignement avec les objectifs de recherche basés sur les embeddings, sans modifier son ton ni son intention globale.
- Supposant que l'utilisateur ne peut pas fournir de clarification, apportez des améliorations basées sur ce que le prompt semble vouloir accomplir.  
- Fournissez uniquement la version améliorée du prompt, en conservant autant que possible le langage original.
"""


class Prompter:
    def __init__(self, llm: ChatOllama):
        self.llm = llm
        self.prompt = PromptTemplate(input_variables=["user_prompt"], template=TEMPLATE)

    def __call__(self, prompt):
        return self.llm.invoke(self.prompt.format(user_prompt=prompt))


if __name__ == "__main__":
    from argparse import ArgumentParser

    from ..utilities.llm_models import get_llm_model_chat

    args = ArgumentParser()
    args.add_argument("--prompt", type=str)
    parse = args.parse_args()

    llm = get_llm_model_chat(temperature=0.7, max_tokens=256)
    prompt = Prompter(llm)
    print(prompt(parse.prompt).content)
