import re
import string
from uuid import uuid4

from keybert import KeyBERT
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import ChatOllama


class ChunkingManager:
    def __init__(self, chunk_size, chunk_overlap, llm: ChatOllama, nb_keywords=3):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.nb_keywords = nb_keywords
        self.llm = llm

        self.init_summarize_prompt()
        self.init_keywords_prompt()
        self.init_cleaner_prompt()

        self.define_text_splitter()

    def init_keywords_prompt(self):
        keywords_prompt_template = """
        Tâche : Identifier les trois mots-clés les plus importants du texte suivant extrait d'un livre.

        Instructions :
        1. Concentrez-vous sur les mots ou expressions qui capturent le mieux les thèmes ou idées principaux du texte.
        2. Évitez d'inclure des termes communs ou génériques, sauf s'ils sont essentiels au contexte.
        3. Fournissez les mots-clés sous forme de liste, séparés par des virgules.

        Texte :
        {paragraph}

        Sortie :
        Mots-clés, séparés par des virgules
        """
        self.keywords_prompt = PromptTemplate(
            input_variables=["paragraph"], template=keywords_prompt_template
        )

    def init_summarize_prompt(self):
        summarize_prompt_template = """
**Tâche :** Résumer l'extrait suivant d'un livre tout en préservant son flux logique, sa cohérence et sa structure de paragraphes.

**Instructions :**  
1. Pour chaque paragraphe, identifiez l'idée principale ou le thème central.  
2. Réécrivez chaque paragraphe de manière concise, en conservant les points clés, les relations entre les concepts et tout contexte nécessaire.  
3. Assurez-vous que le texte résumé est clair, ordonné logiquement et fidèle au contenu original.  
4. N'ajoutez aucun texte supplémentaire, commentaire ou explication — fournissez uniquement le texte résumé.

**Extrait :**  
{paragraph}  

**Livrable :** Fournir une version résumée de l'extrait en suivant ces directives.
"""

        self.summarize_prompt = PromptTemplate(
            input_variables=["paragraph"], template=summarize_prompt_template
        )

    def define_text_splitter(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def init_cleaner_prompt(self):
        template = """
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

        self.cleaner_prompt = PromptTemplate(
            input_variables=["text"], template=template
        )

    def merge_sentences(self, sentence: str):
        punctuation = "?!."
        sentences = sentence.strip().split("\n")
        result = sentences[0].strip()
        for i, j in zip(sentences, sentences[1:]):
            i, j = i.strip(), j.strip()  # Strip leading and trailing whitespace

            # If the line ends with a hyphen, merge without space to continue the word
            if i.endswith("-"):
                result += j[:-1]
            # If the line ends with punctuation, start a new line
            elif i and i[-1] in punctuation:
                result += "\n" + j
            # If both lines start with uppercase, consider it a new paragraph/title
            elif j and i.isupper() and j[0].isupper():
                result += "\n" + j
            # Otherwise, add a space and continue the sentence
            else:
                result += " " + j if j else ""

        return result.strip()

    def is_potential_title(self, line: str) -> bool:
        line = line.strip()  # Remove leading and trailing whitespace

        # Check if the line starts with a number followed by a period (e.g., "19.")
        if re.match(r"^\d+\.", line):
            return True

        # Other heuristics to identify if it's likely a title
        if len(line.split()) <= 10 and not line.endswith(
            tuple(string.punctuation)
        ):  # Short and no ending punctuation
            if line[0].isupper():  # Starts with an uppercase letter
                words = line.split()
                # Optional: Check for capitalized words in the title
                capitalized_words = sum(1 for word in words if word[0].isupper())
                # Allow a few lowercase words if there are capitalized ones
                if capitalized_words >= len(words) / 2:
                    return True

        return False

    def is_valid_file(self, raw: str):
        lines = [i for i in raw.split("\n") if i.strip()]
        count = [i for i in lines if ".............." in i]  # potential title
        count2 = [i for i in lines if "—." in i]  # potential citation
        count3 = [
            1 for i in lines if self.is_potential_title(i)
        ]  # potential citation part
        return (
            (50 > len(lines) > 10)
            and len(count) < 5
            and len(count2) < 5
            and len(count3) < 5
        )

    def clean_text(self, text):
        prompt = self.cleaner_prompt.format(text=text)
        cleaned = self.llm.invoke([("system", prompt)]).content

        return cleaned

    def generate_summaries(self, paragraphs: str):
        summaries = []
        for paragraph in paragraphs:
            input_prompt = self.summarize_prompt.format(paragraph=paragraph)
            summary = self.llm.invoke([("system", input_prompt)]).content
            summaries.append(summary.strip())
        return summaries

    def generate_keywords(self, paragraphs: str, use_llm=True):
        keywords_list = []
        if not use_llm:
            kw_model = KeyBERT()
        for paragraph in paragraphs:
            input_prompt = self.keywords_prompt.format(paragraph=paragraph)
            if use_llm:
                keywords = self.llm.invoke([("system", input_prompt)]).content
                keywords_list.append(keywords.strip().split(","))
            else:
                keywords = kw_model.extract_keywords(
                    paragraph,
                    keyphrase_ngram_range=(1, 2),
                    stop_words="french",
                    top_n=self.nb_keywords,
                )
                keywords = [kw[0] for kw in keywords]
                keywords_list.append(keywords)
        return keywords_list

    def split_text_into_large_chunks(self, text: str, target_word_count=300):
        """
        on splitte le texte en chunks qui vont être résumés par la suite
        """
        paragraphs = [para.strip() for para in text.split("\n") if para.strip()]
        chunks = []
        current_chunk = []
        current_word_count = 0
        for paragraph in paragraphs:
            word_count = len(re.findall(r"\w+", paragraph))
            if current_word_count + word_count >= target_word_count:
                current_chunk.append(paragraph)
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0
            current_chunk.append(paragraph)
            current_word_count += word_count

        # Ajouter le dernier chunk s'il reste des paragraphes
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        return chunks

    def retrieve_documents_from_file(
        self,
        file_path: str,
        use_llm_for_keywords=True,
        summarize_before_chunk=True,
        check_text_validity=True,
        verbose=False,
    ):
        with open(file_path, mode="r") as f:
            text = f.read()

        if verbose:
            print("Raw text:")
            print(text)

        text = self.merge_sentences(text)

        text = self.clean_text(text)

        if check_text_validity:
            if not self.is_valid_file(text):
                print(text)
                print("The text is invalid, not retrieving documents from it.")
                return None

        if verbose:
            print("Cleaned text:")
            print(text)
            print("***********************************\n")

        if summarize_before_chunk:
            large_chunks = self.split_text_into_large_chunks(text)
            summaries = self.generate_summaries(large_chunks)

            if verbose:
                for chunk, summary in zip(large_chunks, summaries):
                    print("Text: ", chunk)
                    print("Summary: ", summary)
                    print("---------------------------------------")

            text = "\n".join(summaries)

        chunks = self.text_splitter.split_text(text)
        keywords_list = self.generate_keywords(chunks, use_llm=use_llm_for_keywords)

        if verbose:
            print("Final chunks and their keywords: ")
            for chunk, keywords in zip(chunks, keywords_list):
                print("Chunk: ", chunk)
                print("Keywords: ", keywords)
                print("----------------------------------------")

        documents = []
        uuids = [str(uuid4()) for _ in range(len(chunks))]
        for id, chunk, keywords in zip(uuids, chunks, keywords_list):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": file_path, "keywords": keywords},
                    id=id,
                )
            )
        return documents


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Generate questions from text files.")
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the folder containing input text files.",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    from ..utilities.llm_models import get_llm_model_chat

    load_dotenv()

    llm = get_llm_model_chat("OLLAMA", temperature=0.1, max_tokens=256)
    chunkingManager = ChunkingManager(chunk_size=300, chunk_overlap=50, llm=llm)

    file_path = args.input_file

    durations = []
    # run one time before testing the different times
    chunkingManager.retrieve_documents_from_file(
        file_path=file_path,
        verbose=False,
        use_llm_for_keywords=True,
        summarize_before_chunk=True,
        check_text_validity=False,
    )

    for use_llm in [True, False]:
        for summ_before_chunk in [True, False]:
            print(
                "***************************************************************************************************"
            )
            print(
                "use_llm_for_keyword: {}; summarize_before_chunk: {}".format(
                    use_llm, summ_before_chunk
                )
            )
            print(
                "***************************************************************************************************"
            )
            start = time.time()
            documents = chunkingManager.retrieve_documents_from_file(
                file_path=file_path,
                verbose=True,
                use_llm_for_keywords=use_llm,
                summarize_before_chunk=summ_before_chunk,
                check_text_validity=False,
            )
            durations.append(
                [
                    "use_llm_for_keyword: {}; summarize_before_chunk: {}".format(
                        use_llm, summ_before_chunk
                    ),
                    round(time.time() - start, 5),
                ]
            )
            print("\n")

    print(durations)
