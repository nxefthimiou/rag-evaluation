"""
rag-eval.py

Preprocess knowledge base data to build a synthetic dataset for use in RAG evaluation.
"""

# TODO:
# Change typehints to comments

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

DATA_HUB['mric'] = (DATA_URL + 'huggingface_doc.csv',
#                   '9239d8ba602c9b243535d22fb0d5a620ec9ebdee2c196f2b761a3a8bd968b843')
					'b8055953f596aaca2b3f0f9d605967e400702459')


def download(url, folder='../data', sha1_hash=None):
	"""Download file to folder and return the local filepath."""
	if not url.startswith('http'):
		url, sha1_hash = DATA_HUB[url]
	os.makedirs(folder, exist_ok=True)
	fname = os.path.join(folder, url.split('/')[-1])
	# Check if hit cache
	if os.path.exists(fname) and sha1_hash:
		sha1 = hashlib.sha1()
		with open(fname, 'rb') as f:
			while True:
				data = f.read(1048576)
				if not data:
					break
				sha1.update(data)
		if sha1.hexdigest() == sha1_hash:
			return fname
	# Download
	print("Downloading {fname} from {url}...")
	r = requests.get(url, stream=True, verify=True)
	with open(fname, 'wb') as f:
		f.write(r.content)
	return fname


def read_dataset(url):
    fname = download(url)
    return datasets.load_dataset("csv", data_files=fname,
                                 split="train", streaming=True)


def preprocess(
	dataset,				# datasets.Dataset
	chunk_size,				# int
	chunk_overlap=None,		# int = None
	batch_size=32,			# int = 32
):							# Iterable[LangchainDocument]
	if chunk_overlap is None:
		chunk_overlap = int(chunk_size / 10)
	
	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=int(chunk_size / 10),
		add_start_index=True,
		separators=["\n\n", "\n", ".", " ", ""],
	)

	for batch in dataset.iter(batch_size):
		for text, source in zip(batch["text"], batch["source"]):
			doc = LangchainDocument(page_content=text, metadata={"source": source})
			docs = text_splitter.split_documents([doc])
			for doc in docs:
				yield doc
			
def data_iter_fn(
	dataset,				# datasets.IterableDataset
	batch_size: int = 32,	# int = 32
):							#  -> Iterable[LangchainDocument]
	"""Generate Langchain documents from a dataset."""
	for batch in dataset.iter(batch_size):	  
		for text, source in zip(batch["text"], batch["source"]):
			yield LangchainDocument(page_content=text, metadata={"source": source})


def split_iter_fn(
	doc_iter,				# List[LangchainDocument], 
	chunk_size,				# int
	chunk_overlap, 			# int = None
#	batch_size=32,			# int = 32
):							# -> Iterable[LangchainDocument]
	"""Generate and return splits of Langchain documents."""
	
	if chunk_overlap is None:
		chunk_overlap = int(chunk_size / 10)

	text_splitter = RecursiveCharacterTextSplitter(
		chunk_size=chunk_size,
		chunk_overlap=int(chunk_size / 10),
		add_start_index=True,
		separators=["\n\n", "\n", ".", " ", ""],
	)
	
	for doc in doc_iter:
		docs = text_splitter.split_documents([doc])
		for doc in docs:
			yield doc


def get_llm_client(
    llm_model,						# str
    timeout = 120,					# Optional[int] = 120
):									# -> InferenceClient
    llm_client = InferenceClient(
		model=llm_model,
		timeout=timeout,
	)
    return llm_client

def get_qa_generation_llm(
    qa_llm_name,					# str
):									# -> InferenceClient
    """Setup QA generation agent."""
    return get_llm_client(config.QA_LLM_MODEL[qa_llm_name])

def get_critique_llm(
    critique_llm_name,				# str 
):									# -> InferenceClient
	"""Setup critique agent."""
	return get_llm_client(config.CRITIQUE_LLM_MODEL[critique_llm_name])


def llm_completion(
	llm_client,						# InferenceClient
	prompt,							# str
	params,							# Dict = {"max_new_tokens": 1000},
):									# -> str
	response = llm_client.post(
		json={
			"inputs": prompt,
			"parameters": params,
			"task": "text-generation",
		},
	)
	return json.loads(response.decode())[0]["generated_text"]


def generate_qa_couples(
	doc_iter,							# Iterable[LangchainDocument]
	qa_agent,							# InferenceClient | Llama
	qa_generation_prompt,				# str
	n_generations,						# int = 10
	max_answer_len,						# int = 300
):										# -> Iterable[Dict]:
	"""Generate QA couples on a LangchainDocument iterator."""
	
	print("Generating {n_generations} QA couples...")
	
	if n_generations > 0:
		doc_iter = random.sample(doc_iter, n_generations)
	
	for doc in doc_iter:
		# generate QA couple
		response = llm_completion(qa_agent,
			qa_generation_prompt.format(context=doc.page_content))
		try:
			question = response.split("Factoid question: ")[-1].split("Answer: ")[0]
			answer = response.split("Answer: ")[-1] 
			assert len(answer) < max_answer_len, "Answer is too long"
			# yield rec
			qa_dict = {
				"context": doc.page_content,
				"question": question,
				"answer": answer,
				"source_doc": doc.metadata["source"],
			}
			yield qa_dict	
		except:
			continue


def generate_critique(
	qa_iter,									# Iterable[Dict]
	critique_agent,								# InferenceClient
	critique_config,							# Dict[criterion, prompt]
	question_groundedness_critique_prompt,		# str
	question_relevance_critique_prompt,			# str
	question_standalone_critique_prompt,		# sstr
):												# -> Iterable[Dict]
	"""Generate critique for QA couples. Modifies in place QA couples records."""
	print("Generating critique for each QA couple...")
	
	for rec in qa_iter:
		evaluations = {
			"groundedness": llm_completion(critique_agent,
				question_groundedness_critique_prompt.format(context=rec["context"],
					question=rec["question"])),
			"standalone": llm_completion(critique_agent,
				question_standalone_critique_prompt.format(question=rec["question"])),
			"relevance": llm_completion(critique_agent,
				question_relevance_critique_prompt.format(question=rec["question"])),
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
				yield rec
		except Exception as e:
			continue



def get_eval_dataset() -> datasets.Dataset:
	"""Generate evaluation dataset for RAG evaluation."""
	raw_dataset = read_dataset('mric') 
	langchain_docs = data_iter_fn(raw_dataset)
	docs_processed = split_iter_fn(langchain_docs)
	
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


if __name__ == '__main__':

	dataset = read_dataset('mric')
	
	chunk_size, chunk_overlap, batch_size = 1000, 100, 256
	chunk_iter = preprocess(dataset, chunk_size, chunk_overlap, batch_size)
	
	# qa_agent = get_model(...)
	qa_iter = generate_qa_couples(chunk_iter, qa_agent, prompts.QA_generation)

	# critique_agent = get_model(...)
	qa_dict = generate_critique(qa_iter, critique_agent, prompts.question_groundedness_critique,
				prompts.question_relevance_critique_prompt, prompts.question_standalone_critique)
): 		
