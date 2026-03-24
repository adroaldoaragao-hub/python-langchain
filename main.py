from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import Field, BaseModel 
from dotenv import load_dotenv
from langchain.globals import set_debug
import os
    
set_debug(True)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

class Destino(BaseModel):
    cidade: str = Field(description="Nome da cidade")
    motivo: str = Field(description="Motivo pelo qual é interessante visitar a cidade")

class Restaurantes(BaseModel):
    cidade: str = Field(description="A cidade recomendada para visitar'")
    restaurantes: str = Field(description="restaurantes recomendados na cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dados o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions() }
)

prompt_restaurantes = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}.
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions() }
)

prompt_cultural = PromptTemplate(
    template="""
    Sugira atividades e locais culturais em {cidade}.
    """,
)

modelo = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=api_key,
    temperature=0.5
)

cadeia_1 = prompt_cidade | modelo | parseador_destino
cadeia_2 = prompt_restaurantes | modelo | parseador_restaurantes
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

cadeia = cadeia_1 | cadeia_2 | cadeia_3

resposta = cadeia.invoke(
    {
        "interesse": "parias"
    }
)
# print("Prompt:\n", prompt_cidade)


print(resposta)