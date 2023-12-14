import argparse

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task", help="Task to perform", required=True)
    parser.add_argument(
        "-l",
        "--language",
        help="Programming language to perform the task in",
        required=True,
    )

    return parser.parse_args()


args = parse_arguments()

llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)
test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "code"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")
test_chain = LLMChain(llm=llm, prompt=test_prompt, output_key="test")

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["code", "test"],
)

result = chain({"language": args.language, "task": args.task})

print("GENERATED CODE:")
print(result["code"])
print("GENERATED TEST:")
print(result["test"])
