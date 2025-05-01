"""
rag.py

Provides functions to setup and use a RAG system to answer questions given a knowledge base,
and to evaluate the performance of the RAG system.
"""

import json
import os
from typing import Dict, Optional, List, Tuple
from tqdm.auto import tqdm

import datasets

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS, VectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.language_models.llms import LLM

from langchain.prompts.chat import (
	ChatPromptTemplate,
	HumanMessagePromptTemplate,
)

from langchain.schema import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFaceHub

from langchain_core.language_models.chat_models import BaseChatModel

from transformers import AutoTokenizer

from ragatouille import RAGPretrainedModel

import config
from data import data_get_documents, get_eval_dataset
import prompts

def split_documents(
	chunk_size: int,
	knowledge_base: List[LangchainDocument],
	tokenizer_name: str,
) -> List[LangchainDocument]:
	"""Split a list of documents into chunks of size `chunk_size` characters."""
	text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
		AutoTokenizer.from_pretrained(tokenizer_name),
		chunk_size=chunk_size,
		chunk_overlap=int(chunk_size / 10),
		add_start_index=True,
		separators=["\n\n", "\n", ".", " ", ""],
	)

	docs_processed = [text_splitter.split_documents([doc]) for doc in knowledge_base]
	
	# remove duplicates
	unique_texts = {}
	docs_processed_unique = []
	for doc in docs_processed:
		if doc.page_content not in unique_texts:
			unique_texts[doc.page_content] = True
			docs_processed_unique.append(doc)
	
	return docs_processed_unique


def load_embeddings(
	langchain_docs: List[LangchainDocument],
	chunk_size: int,
	embedding_model_name: Optional[str] = "thenlper/gte-small",
) -> FAISS:
	"""Returns a FAISS index from the given embedding model and documents."""
	
	# load embedding model
	embedding_model = HuggingFaceEmbeddings(
		model_name=embedding_model_name,
		multi_process=True,
		model_kwargs={"device": "cpu"},
		encode_args={"normalize_embeddings": True},	# set True to compute cosine similarity
	)
	
	# check if embeddings already exist on disk
	index_name = f"index_chunk:{chunk_size}_embeddings:{embedding_model_name.replace(
'/', '~')}"
	index_folder_path = f"./data/indexes/{index_name}"
	if os.path.isdir(index_folder_path):
		return FAISS.load_local(
				index_folder_path,
				embedding_model,
				distance_strategy=DistanceStrategy.COSINE,
		)
	else:
		print("Index not found, generating it...")
		docs_processed = split_documents(
			chunk_size,
			langchain_docs,
			embedding_model_name,
		)
		knowledge_index = FAISS.from_documents(
			docs_processed, embedding_model,
			distance_strategy=DistanceStrategy.COSINE,
		)
		knowledge_index.save_local(index_folder_path)
		return knowledge_index


def get_reader_llm(
	reader_llm_name: str = config.READER_LLM_NAME,
	max_new_tokens: Optional[int] = 512,
	top_k: Optional[int] = 30,
	temperature: Optional[float] = 0.1,
	repetition_penalty: Optional[float] = 1.03,
) -> LLM: 
	"""Reader LLM reads the retrieved documents to formulate its answer."""
	reader_llm = HuggingFaceHub(
		repo_id=config.READER_LLM_REPO[reader_llm_name],
		task="text-generation",
		model_kwargs={
			"max_new_tokens": max_new_tokens,
			"top_k": top_k,
			"temperature": temperature,
			"repetition_penalty": repetition_penalty,
		},
	)
	config_rec = {
		"reader_llm_name": reader_llm_name,
		"max_new_tokens": max_new_tokens,
		"top_k": top_k,
		"temperature": temperature,
		"repetition_penalty": repetition_penalty,
	}
	return reader_llm, config_rec


def rag_chat_completion(
	question: str,
	reader_llm: LLM,
	kb_index: VectorStore,
	reranker: Optional[RAGPretrainedModel] = None,
	num_retrieved_docs: int = 30,
	num_docs_final: int = 7,
) -> Tuple[str, List[LangchainDocument]]:
	"""Answer a question using RAG with the given knwoledge base index."""
	
	# gather documents with retriever
	relevant_docs = kb_index.similarity_search(query=question, k=num_retrieved_docs)
	relevant_docs = [doc.page_content for doc in relevant_docs]		# keep only the text
	
	# optionally rerank results
	if reranker:
		relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
		relevant_docs = [doc["content"] for doc in relevant_docs]
	
	relevant_docs = relevant_docs[:num_docs_final]
	
	# build the final prompt
	context = "\nExtracted documents:\n"
	context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])
	
	final_prompt = prompts.RAG_TEMPLATE.format(question=question, context=context)
	
	# redact an answer
	answer = reader_llm(final_prompt)
	
	return answer, relevant_docs


def run_rag_tests(
	eval_dataset: datasets.Dataset,
	reader_llm: BaseChatModel,
	kb_index: VectorStore,
	output_file: str,
	reranker: Optional[RAGRerankerModel] = None,
	verbose: Optional[bool] = True,
	test_settings: Optional[str] = None,	# used to document test settings
) -> None:
	"""Runs RAG tests on an evaluation dataset and saves the results to a disk file."""
	
	try:	# load previous generations if they exist
		with open(output_file, "r") as f:
			outputs = json.loads(f)
	except:
		outputs = []
	
	for rec in tqdm(eval_dataset):
		question = rec["question"]
		if question in [output["question"] for output in outputs]:
			continue

		answer, relevant_docs = rag_chat_completion(question, reader_llm, kb_index, reranker=reranker)
		if verbose:
			print("==========================================================")
			print(f"Question: {question}")
			print(f"Answer: {answer}")
			print(f"True answer: {rec["answer"]}")
		result = {
			"question": question,
			"true_answer": rec["answer"],
			"source_doc": rec["source"],
			"generated_answer": answer,
			"retrieved_docs": [doc for doc in relevant_docs],
		}
		if test_settings:
			result["test_settings"] = test_settings
		outputs.append(result)
	
	with open(output_file, "w") as f:
		json.dump(outputs, f)


def get_judge_llm(
	temperature: Optional[float] = 0.0,
	judge_llm_name: Optional[str] = "GPT4",
) -> Tuple[BaseChatModel, str]:
	"""Setup judje agent."""
	eval_chat_model = ChatOpenAI(model=config.JUDGE_LLM_MODEL[judge_llm_name], temperature=temperature)
	config_rec = {
		"judge_llm_name": judge_llm_name,
		"temperature": temperature,
	}
	return eval_chat_model, config_rec


def evaluate_answers(
	answer_path: str,
	eval_chat_model: BaseChatModel,
	evaluator_name: str,
	eval_prompt_template: str,
) -> None:
	"""Evaluate generated answers. Modifies the given answer file in place for
	better checkpointing."""
	answers = []
	if os.path.isfile(answer_path):		# load previous generations if they exist
		with open(answer_path, "r") as f:
			answers = json.loads(f)
	
	for experiment in tqdm(answers):
		if f"eval_score_{evaluator_name}" in experiment:
			continue
		eval_prompt = eval_prompt_template.format_messages(
			instruction=experiment["question"],
			response=experiment["generated_answer"],
			reference_answer=experiment["true_answer"],
		)
		eval_result = eval_chat_model.invoke(eval_prompt)
		feedback, score = [item.strip() for item in eval_result.content.split("[RESULT")]
		experiment[f"eval_score_{evaluator_name}"] = score
		experiment[f"eval_feedback_{evaluator_name}"] = feedback
	
	with open(answer_path, "w") as f:
		json.dump(answers, f)



def run_tests_and_evaluate(
    params: Dict = {
		"chunk_size": [2000],
		"embeddings": ["thenlper/gte-small"],
		"rerank": [True, False],
	},
) -> None:
	"""Run RAG tests and evaluate answers."""
	if not os.path.exists("./output"):
		os.mkdir("./output")
	
	reader_llm, reader_model_name = get_reader_llm()
	eval_chat_model, evaluator_name = get_judge_llm()

	evaluation_prompt_template = ChatPromptTemplate.from_messages(
		[
			SystemMessage(content="You are a fair evaluator language model."),
			HumanMessagePromptTemplate.from_template(prompts.EVALUATION),
		]
	)
	
	raw_kb = data_get_documents()
	eval_dataset = get_eval_dataset()

	for chunk_size in params["chunk_size"]:
		for embedding in params["embeddings"]:
			for rerank in params["rerank"]:
				settings_name = f"chunk:{chunk_size}_embeddings:{embedding.replace('/', '~')}"
				settings_name = f"_rerank:{rerank}_reader-model={reader_model_name}.json"
				output_file_name = f"./output/{settings_name}"
				
				print(f"Running evaluation for {settings_name}")
				
				print("Loading knowledge base embeddings...")
				kb_index = load_embeddings(
					raw_kb,
					chunk_size=chunk_size,
					embedding_model_name=embedding,
				)
				
				print("Running RAG...")
				reranker = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0") if rerank else None
				run_rag_tests(
					eval_dataset=eval_dataset,
					llm=reader_llm,
					kb_index=kb_index,
					output_file=output_file_name,
					reranker=reranker,
					verbose=False,
					test_settings=settings_name,
				)
				
				print("Running evaluation...")
				evaluate_answers(
					output_file_name,
					eval_chat_model,
					evaluator_name,
					evaluation_prompt_template,
				)
