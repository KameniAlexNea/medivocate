TEMPLATE = """
Vous êtes un assistant spécialisé dans la reformulation et la décomposition de requêtes utilisateur. Votre tâche consiste à analyser la requête suivante et à effectuer l'une des actions suivantes :

1. Si la requête est complexe (elle contient plusieurs parties ou questions imbriquées), décomposez-la en plusieurs sous-requêtes simples, chacune ciblant un aspect spécifique de la requête.
2. Si la requête est déjà simple et claire, laissez-la telle quelle, mais placez-la dans un tableau.

Dans tous les cas, fournissez les sous-requêtes ou la requête d'origine sous forme d'un tableau, sans ajouter de commentaires ou d'explications.

### Requête utilisateur :
{query}

### Résultat attendu (tableau de sous-requêtes ou de la requête unique) :
[
"Requête 1",
"Requête 2",
...
]
"""
import json

from langchain.prompts import PromptTemplate


class QueryDecomposer:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(input_variables=["query"], template=TEMPLATE)
        self.decomposition_chain = self.prompt | self.llm

    def __call__(self, query):
        sub_queries = self.decomposition_chain.invoke({"query": query}).content

        try:
            index_left = sub_queries.index("[")
            index_right = sub_queries.index("]")
            sub_queries = json.loads(sub_queries[index_left : index_right + 1])
            return sub_queries
        except json.JSONDecodeError:
            print("output format incorrect, query not divided")

        return [query]
