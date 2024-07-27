from haystack.components.evaluators import ContextRelevanceEvaluator

questions = ["Who created the Python Programming Language?"]

contexts = [
    [
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. "
        "Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
    ],
]

evaluator = ContextRelevanceEvaluator()
result = evaluator.run(questions=questions, contexts=contexts)
print(result["score"])
print(result["individual_scores"])
print(result["results"])
