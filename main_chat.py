import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    temperature=0.5
)

prompt_sugestao = ChatPromptTemplate.from_messages([
    ("system", "Você é um guia de viagens especializado em destinos brasileiros. Apresente-se como Sr. Passeios"),
    ("placeholder", "{historico}"),
    ("human", "{query}")
])

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "aula_langchain_alura"

def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

lista_perguntas = [
    "Quero visitar um lugar no Brasil, famoso por praias e culturas. Pode sugerir?",
    "Qual a melhor época para visitar esse lugar?"
]

cadeia_com_memoria= RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)

for pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke(
        {"query": pergunta},
        {"configurable": {"session_id": sessao}}
    )
    print(f"Pergunta: {pergunta}")
    print(f"Resposta: {resposta}\n")