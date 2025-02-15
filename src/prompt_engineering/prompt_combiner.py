TEMPLATE = """
Le prompt suivant a été rentré:
{prompt}
Vous avez décomposé ce prompt et répondu à chaque sous requête.
Vous avez récupéré les informations suivantes pour chaque sous-requête :
{results}

Combinez ces informations pour produire une réponse claire, cohérente et complète au prompt posé en entré.
"""


from langchain.prompts import PromptTemplate


class PromptCombiner:
    def __init__(self, llm):
        self.llm = llm
        self.prompt = PromptTemplate(
            input_variables=["prompt", "results"], template=TEMPLATE
        )
        self.combiner_chain = self.prompt | self.llm

    def __call__(self, prompt, results):
        for token in self.combiner_chain.stream({"prompt": prompt, "results": results}):
            yield token.content
