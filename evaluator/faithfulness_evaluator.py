from haystack.components.evaluators import FaithfulnessEvaluator

questions = ["Who created Python Programming Language?"]
contexts = [
    [
        "Python, created by Guido van Rossum in the late 1980s, is a high-level general-purpose programming language. "
        "Its design philosophy emphasizes code readability, and its language constructs aim to help programmers write clear, logical code for both small and large-scale software projects."
    ],
]

predicted_answers = ["Python is a high-level general-purpose programming language that was created by Guido van Rossum"]
evaluator = FaithfulnessEvaluator()
result = evaluator.run(questions=questions, contexts=contexts, predicted_answers=predicted_answers)

print(result["individual_scores"])

print(result["results"])