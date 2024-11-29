import os

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq

groq_api_key = os.getenv('API_KEY')

# Initialize the ChatGroq LLM with your API key and model.
llm = ChatGroq(
    groq_api_key='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',
    model_name='llama3-70b-8192'
)

# Define your data structure for the response, now expecting a boolean for the setup field.
class Joke(BaseModel):
    setup: bool = Field(description="True if it is a Job description, and False if Not a Job Description")

# Example joke query
joke_query = '''
This output parser allows users to specify an arbitrary JSON schema and query LLMs for outputs that conform to that schema.
Keep in mind that large language models are leaky abstractions! You'll have to use an LLM with sufficient capacity to generate well-formed JSON.
'''

# Set up the parser with the boolean field in the response.
parser = JsonOutputParser(pydantic_object=Joke)

# Define the prompt template that includes the instructions and query.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Chain the prompt, LLM, and parser.
chain = prompt | llm | parser

# Loop for invoking the chain and print the result for each iteration.
for i in range(1, 10):
    response = chain.invoke({"query": joke_query})
    print(response.setup)  # This will output True or False
