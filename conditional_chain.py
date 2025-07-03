from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from typing import Literal


load_dotenv()

model = ChatOpenAI()



class preview(BaseModel):
    sentiment:Literal["pos","neg"]= Field(description='Give the sentiment of the feedback')


parser1=PydanticOutputParser(pydantic_object=preview)

parser2=StrOutputParser()



prompt1=PromptTemplate(
    template="Classify the sentiment of the following feedback text into postive or negative \n {feedback}\n{format}" ,
    input_variables=['feedback'],
    partial_variables={"format":parser1.get_format_instructions()}
)


prompt2=PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=['feedback']
)

con_chain=RunnableBranch(
    (lambda x:x.sentiment=="pos",prompt2|model|parser2),
    (lambda x:x.sentiment=="neg",prompt3|model|parser2),
    RunnableLambda(lambda x:"couldnt find sentiment")
)


chain=prompt1|model|parser1|con_chain

res=chain.invoke({'feedback': 'This is a beautiful phone'})
print(res)

chain.get_graph().draw_ascii()