import os
import hashlib
import requests

from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import datasets

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument

from huggingface_hub import InferenceClient

import config

DATA_HUB = dict()
DATA_URL = 'https://huggingface.co/datasets/m-ric/huggingface_doc/resolve/main/'

DATA_HUB['mric'] = (DATA_URL + 'huggingface_doc.csv',
                               '9239d8ba602c9b243535d22fb0d5a620ec9ebdee2c196f2b761a3a8bd968b843')

def _download(url, folder='../data', sha256_hash=None):
    """Download file to folder and return the local path."""
    if not url.startswith('http'):
        url, sha256_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha256_hash:
        sha256 = hashlib.sha256()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha256.update(data)
        if sha256.hexdigest() == sha256_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname


## Load knowledge base
def read_mric_dataset(url):
    fname = _download(url)
    ds = datasets.load_dataset("csv", datafiles=fname, split="train")
    return ds

# Build a synthetic dataset for evaluation

## Prepare source documents

def dataset_doc_iter(
    dataset,            # datasets.Dataset
):                      # Iterable[LangchainDocument]
    """Yields Langchain documents from the records of a dataset."""
    for rec in dataset:
        yield LangchainDocument(page_content=rec["text"], metadata={"source":
                    rec["source"]})


def split_doc_iter(
    doc_iter,               # Iterable[LangchainDocument]
    chunk_size,             # int
    chunk_overlap=None,     # int = None
):
    """Splits LangchainDocument's from an iterable into chunks and yields the chunks."""
    if chunk_overlap is None:
        chunk_overlap = chunk_size / 10
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    for doc in doc_iter:
        yield text_splitter.split_documents([doc])


## Setup agents for question generation

def get_huggingface_llm_client(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",     # str
    timeout=120,                                        # int
):                                                      # -> InferenceClient
    llm_client = InferenceClient(
        model=repo_id,
        timeout=timeout,
    )
    return llm_client

def call_llm(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]


def qa_generation(
    llm_client,                 # InferenceClient
    doc_iter,                   # Iterable[LangchainDocument]
    #n_generations = 10,
    qa_generation_prompt,       # str
):                              # Iterable[Dict[str, str]]
    #printf(f"Generating {n_generations} QA couples...")
    
    for doc in doc_iter:
        # Generate QA couple
        qa_generated = call_llm(llm_client,
            qa_generation_prompt.format(context=doc.page_content))
        try:
            question = qa_generated.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = qa_generated.split("Answer: ")[-1]
            qa_dict = {
                "context": doc.page_content,
                "question": question,
                "answer": answer,
                "source_doc": doc.metadata["source"],
            }
            yield qa_dict
        except:
            continue


## Setup critique agents

def critique_generation(
    llm_client,                                         # InferenceClient
    qa_dict_iter,                                       # Iterable[Dict]
    question_groundedness_critique_prompt,              # str
    question_relevance_critique_prompt,                 # str
    question_standalone_critique_prompt,                # str
):
    print("Generating critique for QA couple...")
    for qa_dict in qa_dict_iter:
        evaluations = {
            "groundedness": call_llm(
                llm_client,
                    question_groundedness_critique_prompt.format(context=qa_dict["context"],
                    question=qa_dict["question"]),
            ),
            "relevance": call_llm(
                llm_client,
                question_relevance_critique_prompt.format(context=qa_dict["context"]),
            ),
            "standalone": call_llm(
                llm_client,
                question_standalone_critique_prompt.format(context=qa_dict["context"]),
            ),
        }
        try:
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()),
                    evaluation.split("Total rating: ")[-2].split("Evaluation :")[1],
                )
                qa_dict.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
        except Exception as e:
            continue

### Now filter out bad questions

def get_evaluation_dataset(
    critique_iter,              # Iterable[Dict]
):                              # datasets.dataset

    questions_with_critique = pd.DataFrame.from_dict(critique_iter)
    questions_filtered = questions_with_critique.loc[
        (questions_with_critique["groundedness_score"] >= 4)
        & (questions_with_critique["relevance_score"] >= 4)
        & (questions_with_critique["standalone_score"] >= 4)
    ]

    eval_dataset = datasets.Dataset.from_pandas(questions_filtered, split="train",
        preserve_index=False)
    return eval_dataset


# if __name__ == "__main__":
    
#     dataset = read_mric_dataset('mric')
#     doc_iter = dataset_doc_iter(dataset)
#     chunk_size = 2000
#     doc_iter = split_doc_iter(doc_iter, chunk_size)
#     qa_agent = get_huggingface_llm_client()
#     qa_iter = qa_generation(qa_agent, doc_iter, config.QA_generation_prompt)
#     critique_iter = critique_generation(qa_agent, qa_iter,
#         config.question_groundedness_critique_prompt,
#         config.question_relevance_critique_prompt,
#         config.question_standalone_critique_prompt,
#     )
#     #eval_dataset = get_evaluation_dataset(critique_iter)

