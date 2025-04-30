
import os
from pathlib import Path
from llama_cpp import Llama

def get_model(
    model_hub: str,
    model_name: str,
    model_file: str,
    n_ctx: int = 16000,
) -> Llama:
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

    # Run inference
    #prompt = "What a Kalman filter is?"
    resp = llm(prompt, **generation_kwargs)  # res is a dictionary
    return resp["choices"][0]["text"]

if __name__ == "__main__":
    # Organization of models into disk files
    # Each model file is stored in `model_hub`/`model_name`\`model_file`, where:
    # `model_hub` is the "root" directory where all model files live
    # `model_name` is the name of the model
    # `model_file` is the name of the model file
    model_hub = "C:/Users/efthimiou/AppData/Roaming/Jan/data/models/huggingface.co/TheBloke/"
    model_name = "zephyr-7B-beta-GGUF"
    model_file = "zephyr-7b-beta.Q2_K.gguf"
    
    llm = get_model(model_hub, model_name, model_file)

    # Generation kwargs
    generation_kwargs = {
        "max_tokens": 2000,
        "stop": ['</s>'],
        "echo": False,
        "top_k": 40,
        "top_p": 0.95,
        "temperature": 0.7,
    }

    # Run inference
    prompt = "What is the capital of Greece?"
    res = llm_completion(llm, prompt, generation_kwargs)
    print(res)

