import textwrap
import dspy


class EvalSignature(dspy.Signature):
    __doc__ = textwrap.dedent("""
    My language model wrote a pytorch program to train a network to high accuracy in short time.
    However, it didn't mange to do so. I need you to analyze the output and explain what went wrong.
    """)

    program: str = dspy.InputField(desc="The program that failed")
    intended_score: float = dspy.InputField(desc="The accuracy the model should get")
    analysis: str = dspy.InputField(desc="The output of the model")
    explanation: str = dspy.InputField(desc="The validation error trigged by the models output")
    evaluation: str = dspy.OutputField(desc="Explain what the model did wrong")


def model_eval(args, to_worker, from_worker, worker_idx):
    predictor = dspy.TypedPredictor(EvalSignature)
    while True:
        program, intended_score, analysis, explanation = to_worker.get()
        pred = predictor.predict(
            program=program,
            intended_score=intended_score,
            analysis=analysis,
            explanation=explanation,
        )
        from_worker.put((pred.evaluation,))
