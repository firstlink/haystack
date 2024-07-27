from haystack.components.evaluators import SASEvaluator
sas_evaluator = SASEvaluator()
sas_evaluator.warm_up()
result = sas_evaluator.run(ground_truth_answers=["Berlin", "Paris"], predicted_answers=["Berlin", "Lyon"])
print(result["individual_scores"])
print(result["score"])
