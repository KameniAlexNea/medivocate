from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama


class BaseAgent:
    def __init__(self, llm: ChatOllama, prompt: str):
        self.llm = llm
        self.prompt_template = PromptTemplate(
            input_variables=["inputs"], template=prompt
        )

    def __call__(self, text):
        input_prompt = self.prompt_template.format(inputs=text)
        return self.llm.invoke([("system", input_prompt)])
    
    def batch_process(self, texts: list[str]):
        return [self.process(text) for text in texts]

    def process(self, text: str):
        return self.__call__(text)


class SummaryAgent(BaseAgent):
    def __init__(self, llm):
        summarize_prompt_template = """
**Tâche :** Résumer l'extrait suivant d'un livre tout en préservant son flux logique, sa cohérence et sa structure de paragraphes.

**Instructions :**  
1. Pour chaque paragraphe, identifiez l'idée principale ou le thème central.  
2. Réécrivez chaque paragraphe de manière concise, en conservant les points clés, les relations entre les concepts et tout contexte nécessaire.  
3. Assurez-vous que le texte résumé est clair, ordonné logiquement et fidèle au contenu original.  
4. N'ajoutez aucun texte supplémentaire, commentaire ou explication — fournissez uniquement le texte résumé.

**Extrait :**  
{inputs}  

**Livrable :** Fournir une version résumée de l'extrait en suivant ces directives.
"""
        super().__init__(llm, summarize_prompt_template)


class CleanAgent(BaseAgent):
    def __init__(self, llm):
        summarize_prompt_template = """
**Tâche :** Nettoyer et préparer le texte d'entrée pour une utilisation optimale dans un modèle de génération augmentée par récupération (RAG).

**Instructions :**  
1. Corrigez toutes les fautes d'orthographe tout en préservant le sens original.  
2. Identifiez et séparez les mots collés en leur forme correcte, en veillant à respecter le contexte.  
3. Supprimez les caractères ou symboles inutiles qui pourraient altérer les performances du modèle, sans modifier le contenu essentiel.  
4. Assurez-vous que le texte final est propre, précis et prêt à être traité par le modèle RAG.  
5. Dans votre sortie, la fin d'un paragraphe est définie par une nouvelle ligne.

**Texte d'entrée :**  
{text}

**Sortie (Texte Nettoyé) :**  
[Fournissez uniquement le texte nettoyé et corrigé.]
"""
        super().__init__(llm, summarize_prompt_template)
