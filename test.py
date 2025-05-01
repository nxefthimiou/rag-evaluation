from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI, OpenAI

import prompts

if __name__ == '__main__':
    
    load_dotenv()
    print("API Key:", os.getenv("OPENAI_API_KEY"))
    #assert False, "Ok?"
    # eval_chat_model = ChatOpenAI(model="gpt-4.1", temperature=0)
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "What is the capital of Greece?"}
    # ]
    # resp = eval_chat_model.invoke(messages)
    # # print(type(resp))
    # # print(resp)
    # print(resp.content)
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "What a Kalman filter is?"}
    # ]
    # resp = eval_chat_model.invoke(messages)
    # print(resp.content)
    
    context = """
    Nikolaos E. was born in Trygona, a small village in a mountain area of Central Greece.
    His father was Charalambos who died on 2007 and his mother was Maria who died on Marh, 2024.
    """
    question = "Where was Nikolaos E. born?"
    answer = "Nikolaos E. was born in Trygona, a small village in a mountain area of Central Greece."
    #prompt = prompts.RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    # prompt = prompts.QA_generation.format(context=context) 
    llm_model = OpenAI()
    # resp = llm_model.invoke(prompt)
    # answer = resp.split('Answer: ')[-1]
    
    prompt = prompts.question_groundedness_critique.format(question=question, context=context,
                                                           answer=answer)
    resp = llm_model.invoke(prompt, temperature=0.1)
    print(f'[{resp}]')
    print("\n\n")
    print(prompt)
    
