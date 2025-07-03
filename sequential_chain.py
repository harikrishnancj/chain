from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template='decribe about the {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='sumarize  int 3 point  {text}',
    input_variables=['text']
)
parser=StrOutputParser()
chain=prompt1|model|parser|prompt2|model|parser

res=chain.invoke({"topic":"Apple"})

print(res)

chain.get_graph().print_ascii()