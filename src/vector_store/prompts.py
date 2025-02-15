from langchain_core.prompts.prompt import PromptTemplate

DEFAULT_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant tasked with generating alternative versions of a given user question. Your goal is to create 3 different perspectives on the original question to help retrieve relevant documents from a vector database. This approach aims to overcome some limitations of distance-based similarity search.

When generating alternative questions, follow these guidelines:
1. Maintain the core intent of the original question
2. Use different phrasing, synonyms, or sentence structures
3. Consider potential related aspects or implications of the question
4. Avoid introducing new topics or drastically changing the subject matter

Here is the original question:

{question}

Generate 3 alternative versions of this question. Provide your output as a numbered list, with each alternative question on a new line. Do not include any additional explanation or commentary.

Remember, the purpose of these alternative questions is to broaden the search scope while staying relevant to the user's original intent. This will help in retrieving a diverse set of potentially relevant documents from the vector database.

Do not include any additional explanation or commentary, just give 3 alternative questions.
""",
)