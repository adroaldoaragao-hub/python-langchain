from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import os
from dotenv import load_dotenv
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio

load_dotenv()

modelo = ChatOpenAI(
    model="gpt-4o-mini",
    api_key = os.getenv("OPENAI_API_KEY"),
    temperature=0.5
)

prompt_consultor_praia = ChatPromptTemplate.from_messages([
    ("system", "Apresentes-se como Sra. Praia. Você é uma consultora de viagens especializada em destinos de praia."),
    ("human", "{query}")
])

prompt_consultor_montanha = ChatPromptTemplate.from_messages([
    ("system", "Apresentes-se como Sr. Montanha. Você é uma consultora de viagens especializada em destinos de montanha e atividades radicais."),
    ("human", "{query}")
])

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Responda apenas com 'praia' ou 'montanha'."),
    ("human", "{query}")
])  

roteador = prompt_roteador | modelo.with_structured_output(Rota)

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

async def no_roteador(estado: Estado, config=RunnableConfig()):
    return {
        "destino": await roteador.ainvoke({"query": estado["query"]}, config)
        }

async def no_praia(estado: Estado, config=RunnableConfig()):
    resposta = await cadeia_praia.ainvoke({"query": estado["query"]}, config)
    return {"resposta": resposta}

async def no_montanha(estado: Estado, config=RunnableConfig()):
    resposta = await cadeia_montanha.ainvoke({"query": estado["query"]}, config)
    return {"resposta": resposta}

def escolher_no(estado: Estado) -> Literal["praia", "montanha"]:
    if estado["destino"]["destino"] == "praia":
        return "praia"
    else:
        return "montanha"
    
grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    consulta = "Quero visitar um lugar no Brasil famoso por praias e cultura."
    resposta = await app.ainvoke({"query": consulta})
    print("Resposta final:", resposta["resposta"])

asyncio.run(main())