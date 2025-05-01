
import os
from pathlib import Path
from llama_cpp import Llama

import prompts

def get_model(
    model_hub,        # str
    model_name,        # str
    model_file,        # str
    n_ctx=16000,        # int = 16000
):                    # -> Llama
    model_path = str(Path(model_hub, model_name, model_file))

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,          # context length
        n_threads=2,
        n_gpu_layers=0,
    )
    return llm


def llm_completion(
    llm,                    # Llama
    prompt,                 # str
    generation_kwargs,      # params: Dict = {"max_new_tokens": 1000},
):                          # -> str
    # Generation kwargs
    # generation_kwargs = {
    #     "max_tokens": 2000,
    #     "stop": ['</s>'],
    #     "echo": False,
    #     "top_k": 1,
    # }

    resp = llm(prompt, **generation_kwargs)  # res is a dictionary
    return resp["choices"][0]["text"]

if __name__ == "__main__":
    # Organization of models into disk files
    # Each model file is stored in `model_hub`/`model_name`\`model_file`, where:
    # `model_hub` is the "root" directory where all model files live
    # `model_name` is the name of the model
    # `model_file` is the name of the model file
    model_hub = "/home/nikolaos/.config/Jan/data/models/huggingface.co/TheBloke"
    model_name = "Llama-2-7B-Chat-GGUF"
    model_file = "llama-2-7b-chat.Q2_K.gguf"
    
    llm = get_model(model_hub, model_name, model_file)

    # Generation kwargs
    generation_kwargs = {
        "max_tokens": 2000,
        "stop": ['</s>'],
        "echo": False,
        "top_k": 40,
        "top_p": 0.95,
        "temperature": 0.1,
    }

    #--- Run inference
    context = """
    Nikolaos E. was born in Trygona, a small village in a mountain area of Central Greece.
    His father was Charalambos who died on 2007 and his mother was Maria who died on Marh, 2024.
    """
    question = "Where was Nikolaos E. born?"
    # answer = "Nikolaos E. was born in Trygona, a small village in a mountain area of Central Greece."

    prompt = prompts.question_groundedness_critique.format(context=context, question=question) 
    #--- prompt = "What a Kalman filter is?"
    res = llm_completion(llm, prompt, generation_kwargs)
    print(res)
    
    #--- RAG
    print("answering question with RAG...")
    prompt = prompts.RAG_PROMPT_TEMPLATE.format(context=context, question=question)
    resp = llm_completion(llm, prompt, generation_kwargs)
    print(resp)
