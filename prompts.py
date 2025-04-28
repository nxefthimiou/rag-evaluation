QA_generation = """
Your task is to write a factoid question and answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual
information from the context.
Your factoid question should be formulated in the same style as questions users could
ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to
the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""

question_groundedness_critique = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given
question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not
answerable at all given the context, and 5 means that the question is clearly and
unambiguously answered with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and the context.

Question: {question}\n
Context: {context}\n
Answer:::"""

question_relevance_critique = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be
to machine learning developers building NLP applications with Hugging Face
ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful
at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question:

Question: {question}\n
Answer:::"""

question_standalone_critique = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independant this
question is.
Give your answer in a scale of 1 to 5, where 1 means that the question depends on
additional information to be understood, and 5 means that the question makes sense by
itself.
For instance, if the question refers to a particular setting, like 'in the context'
or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub,
Hugging Face or Space and still be a 5: it must be simply clear to an operatot with
access to the documentation what the question is about.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question:

Question: {question}\n
Answer:::"""

RAG_PROMPT_TEMPLATE = """
<|system|>
Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the
question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.</s>
<|user|>
Context:
{context}
---
Now here is the question you need to answer.

Question: {question}
</s>
<|assistant|>
"""

EVALUATION = """
An instruction (might include an Input inside it), a response to evaluate, a
reference answer that gets a score of 5, and a score rubric representing a evaluation
criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based
on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You
should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for
criteria}} [RESULT] {{an integer between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to
include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference answer (score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely

###Feedback:"""

