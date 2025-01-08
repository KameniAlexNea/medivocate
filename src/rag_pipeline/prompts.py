from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

system_template = """
Vous êtes un assistant IA qui fournit des informations sur l'histoire de l'Afrique et la médecine traditionnelle africaine. Vous recevez une question et fournissez une réponse claire et structurée. Lorsque cela est pertinent, utilisez des points et des listes pour structurer vos réponses.

Utilisez uniquement les éléments de contexte suivants pour répondre à la question de l'utilisateur. Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.

Si la question posée est dans une langue parlée en Afrique ou demande une traduction dans une de ces langues, répondez que vous ne savez pas et demandez à l'utilisateur de reformuler sa question.

Si vous connaissez la réponse à la question mais que cette réponse ne provient pas du contexte ou n'est pas relative à l'histoire africaine ou à la médecine traditionnelle, répondez que vous ne savez pas et demandez à l'utilisateur de reformuler sa question.

-----------------
{context}
"""

messages = [
    MessagesPlaceholder(variable_name="chat_history"),
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{input}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


contextualize_q_system_prompt = (
    "Étant donné un historique de conversation et la dernière question de l'utilisateur "
    "qui pourrait faire référence au contexte dans l'historique de conversation, "
    "formulez une question autonome qui peut être comprise "
    "sans l'historique de conversation. NE répondez PAS à la question, reformulez-la "
    "si nécessaire, sinon retournez-la telle quelle."
)

CONTEXTUEL_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)
