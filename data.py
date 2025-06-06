"""
data.py

Preprocess data to build a synthetic dataset for use in RAG evaluation.
"""

import os
import hashlib
import requests
from tqdm.auto import tqdm
from typing import Optional, List, Tuple, Dict
import json
import random

import datasets

import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.docstore.document import Document as LangchainDocument

from huggingface_hub import InferenceClient

import config
import prompts

DATA_HUB = dict()
DATA_URL = 'https://huggingface.co/datasets/m-ric/huggingface_doc/resolve/main/'

DATA_HUB['huggingface_doc'] = (DATA_URL + 'huggingface_doc.csv',
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
 

def read_dataset(
    url,                # str
    split="train",      # str
    streaming=False,    # bool
):                      # datasets.Dataset
    fname = _download(url)
    return datasets.load_dataset("csv", data_files=fname, split=split,
                                streaming=False)

def data_iter(
    dataset,        # datasets.Dataset,
):                  # -> LangchainDocument
    
    for rec in dataset:
        yield LangchainDocument(page_content=rec["text"],
		metadata={"source": rec["source"]})

def split_documents(
	langchain_docs: List[LangchainDocument],
	chunk_size: int = 2000,
) -> List[LangchainDocument]:
	"""Split a dataset into chunks of size `chunk_size`."""
	
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=int(chunk_size / 10),
		add_start_index=True,
		separators=["\n\n", "\n", ".", " ", ""],
	)
	
	for doc in langchain_docs:
		yield text_splitter.split_documents([doc])


def get_llm_client(
    llm_model: str,
    timeout: Optional[int] = 120,
) -> InferenceClient:
    llm_client = InferenceClient(
		model=llm_model,
		timeout=timeout,
	)
    return llm_client

def get_qa_generation_llm(
    qa_llm_name: str,
) -> InferenceClient:
    """Setup QA generation agent."""
    return get_llm_client(config.QA_LLM_MODEL[qa_llm_name])

def get_critique_llm(
    critique_llm_name: str, 
) -> InferenceClient:
	"""Setup critique agent."""
	return get_llm_client(config.CRITIQUE_LLM_MODEL[critique_llm_name])


def llm_completion(
	llm_client: InferenceClient,
	prompt: str,
	params: Dict = {"max_new_tokens": 1000},
) -> str:
	response = llm_client.post(
		json={
			"inputs": prompt,
			"parameters": params,
			"task": "text-generation",
		},
	)
	return json.loads(response.decode())[0]["generated_text"]


def generate_qa_couples(
	docs_processed: List[LangchainDocument],
	qa_agent: InferenceClient,
	n_generations: int = 10,
	max_answer_len: int = 300,
) -> List[Dict]:
	"""Generate QA couples on a given context."""
	
	print("Generating {n_generations} QA couples...")
	
	qa_records = []
	for doc in random.sample(docs_processed, n_generations):
		# generate QA couple
		response = llm_completion(qa_agent,
			prompts.QA_generation.format(context=doc.page_content))
		try:
			question = response.split("Factoid question: ")[-1].split("Answer: ")[0]
			answer = response.split("Answer: ")[-1] 
			assert len(answer) < max_answer_len, "Answer is too long"
			# yield rec
			qa_records.append(
				{
					"context": doc.page_content,
					"question": question,
					"answer": answer,
					"source_doc": doc.metadata["source"],
				}
			)
		except:
			continue
	return qa_records


def generate_critique(
	qa_records: List[Dict],
	critique_agent: InferenceClient,
) -> None:
	"""Generate critique for QA couples. Modifies in place QA couples records."""
	print("Generating critique for each QA couple...")
	
	for rec in tqdm(qa_records):
		evaluations = {
			"groundedness": llm_completion(critique_agent,
				prompts.question_groundedness_critique.format(context=rec["context"],
					question=rec["question"])),
			"standalone": llm_completion(critique_agent,
				prompts.question_standalone_critique.format(question=rec["question"])),
			"relevance": llm_completion(critique_agent,
				prompts.question_relevance_critique.format(question=rec["question"])),
		}
		try:
			for criterion, evaluation in evaluations:
				score, eval = (
					int(evaluation.split("Total rating: ")[-1].strip()),
					evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1]
				)
				rec.update(
					{
						f"{criterion}_score": score,
						f"{criterion}_eval": eval,
					}
				)
		except Exception as e:
			continue



def get_eval_dataset() -> datasets.Dataset:
	"""Generate evaluation dataset for RAG evaluation."""
	raw_dataset = read_dataset('huggingface_doc') 
	langchain_docs = data_iter(raw_dataset)
	docs_processed = split_documents(langchain_docs)
	
	qa_agent = get_llm_client(config.QA_LLM_MODEL[config.QA_LLM_NAME])
	qa_records = generate_qa_couples(
		docs_processed,
		qa_agent,
	)
	
	critique_agent = get_llm_client(config.CRITIQUE_LLM_MODEL[config.CRITIQUE_LLM_NAME])
	generate_critique(qa_records, critique_agent)
 
	qa_records_df = pd.DataFrame.from_dict(qa_records)
	
	qa_records_df = qa_records_df.loc[
		(qa_records_df["groundedness_score"] >= 4)
		& (qa_records_df["relevance_score"] >= 4)
		& (qa_records_df["standalone_score"] >= 4)
	]

	eval_dataset = datasets.Dataset.from_pandas(qa_records_df, split="train",
										preserve_index=False)
	return eval_dataset



