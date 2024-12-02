# NOTE: copied from llm_rag.
from functools import cache

import numpy as np
from langchain.evaluation import load_evaluator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResult, EvaluationResults
from langsmith.schemas import Example, Run
from rouge_score import rouge_scorer


class FaithfullnessMetrics(RunEvaluator):
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.evaluator_faithfullness = load_evaluator(
            "labeled_score_string",
            criteria={
                "faithful": "How faithful is the submission to the reference context?",
            },
            llm=self.llm,
            normalize_by=10,
        )

    def evaluate_run(self, run: Run, example: Example) -> dict:
        res = self.evaluator_faithfullness.evaluate_strings(
            prediction=next(iter(run.outputs.values())),
            input=run.inputs["Question"],
            reference=example.inputs["documents"],
        )
        return EvaluationResult(key="labeled_criteria:faithful", **res)


class Accuracy(RunEvaluator):
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self.evaluator_accuracy = load_evaluator(
            "labeled_score_string",  # type: ignore
            criteria={
                "accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor errors or omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.""",
            },
            llm=self.llm,
            normalize_by=10,
        )

    def evaluate_run(self, run: Run, example: Example) -> EvaluationResult | EvaluationResults:
        res = self.evaluator_accuracy.evaluate_strings(
            prediction=next(iter(run.outputs.values())),
            input=run.inputs["Question"],
            # We are treating the documents as the reference context in this case.
            reference=example.outputs["Answer"],
        )
        return EvaluationResult(key="labeled_criteria:accuracy", **res)


class CosineSimilarity(RunEvaluator):
    def __init__(self) -> None:
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="thenlper/gte-base",
        )

    def evaluate_run(self, run: Run, example: Example | None = None):
        response = run.outputs["output"]
        reference = example.outputs["Answer"]

        response_embedding = np.array(self.embedding_model.embed_query(response))
        reference_embedding = np.array(self.embedding_model.embed_query(reference))

        dot_product = np.dot(response_embedding, reference_embedding)
        cosine_similarity = dot_product / (np.linalg.norm(response_embedding)) * (np.linalg.norm(reference_embedding))
        return EvaluationResult(
            **{
                "key": "cosine_similarity",
                "score": cosine_similarity,
            }
        )


class RougeLsum(RunEvaluator):
    def __init__(self) -> None:
        # "ROUGE-Lsum splits the text into sentences based on newlines
        # and computes the LCS for each pair of sentences and take the average score for all sentences
        self.scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        self.score_func = cache(self.scorer.score)

    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rougeLsum_f1_score",
                "score": rouge_score["rougeLsum"].fmeasure,
                "comment": f"precision:{rouge_score['rougeLsum'].precision}, recall:{rouge_score['rougeLsum'].recall}",
            },
        )

        return result


class RougeLsumPrecision(RougeLsum):
    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rougeLsum_precision",
                "score": rouge_score["rougeLsum"].precision,
            },
        )
        return result


class RougeLsumRecall(RougeLsum):
    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rougeLsum_recall",
                "score": rouge_score["rougeLsum"].recall,
            },
        )
        return result


class Rouge1(RunEvaluator):
    def __init__(self) -> None:
        # "ROUGE-Lsum splits the text into sentences based on newlines
        # and computes the LCS for each pair of sentences and take the average score for all sentences
        self.scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        self.score_func = cache(self.scorer.score)

    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rouge1_f1_score",
                "score": rouge_score["rouge1"].fmeasure,
                "comment": f"precision:{rouge_score['rouge1'].precision}, recall:{rouge_score['rouge1'].recall}",
            },
        )
        return result


class Rouge1Precision(Rouge1):
    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rouge1_precision",
                "score": rouge_score["rouge1"].precision,
            },
        )
        return result


class Rouge1Recall(Rouge1):
    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rouge1_recall",
                "score": rouge_score["rouge1"].recall,
            },
        )
        return result


class Rouge2(RunEvaluator):
    def __init__(self) -> None:
        # "ROUGE-Lsum splits the text into sentences based on newlines
        # and computes the LCS for each pair of sentences and take the average score for all sentences
        self.scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
        self.score_func = cache(self.scorer.score)

    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rouge2_f1_score",
                "score": rouge_score["rouge2"].fmeasure,
            },
        )
        return result


class Rouge2Precision(Rouge2):
    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rouge2_precision",
                "score": rouge_score["rouge2"].precision,
            },
        )
        return result


class Rouge2Recall(Rouge2):
    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult | EvaluationResults:
        response = run.outputs["output"]
        reference = example.outputs["Answer"]
        rouge_score = self.scorer.score(target=reference, prediction=response)

        result = EvaluationResult(
            **{
                "key": "rouge2_recall",
                "score": rouge_score["rouge2"].recall,
            },
        )
        return result
