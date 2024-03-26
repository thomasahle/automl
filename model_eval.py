import textwrap
import traceback
import dspy


class EvalSignatureOld(dspy.Signature):
    __doc__ = textwrap.dedent("""
    You are an expert in machine learning with speciality in debugging and optimizing neural networks.
    My language model wrote a pytorch program to train a network to high accuracy in short time.

    Summarize the output of the model, making sure to include
    - The number of epochs the model was able to complete within the time limit
    - The speed of the model in terms of epochs per second
    - The training behavior of the model, such as
        - The stability of the training
        - The convergence of the model
        - The overfitting or underfitting of the model
    
    Did the program succeed in the goals of the original plan for the program?

    Analyze what parts of the program contributed to these behaviors.
    How can the program be changed to improve these behaviors?
    """)

    plan: str = dspy.InputField(desc="The intention for the program")
    program: str = dspy.InputField(desc="The program that was run")
    stdout: str = dspy.InputField(desc="The output of the program")
    summary: str = dspy.OutputField(desc="Two short paragraphs")


class EvalSignature(dspy.Signature):
    __doc__ = textwrap.dedent("""
    You are an expert in machine learning with speciality in debugging and optimizing neural networks.
    My language model wrote a PyTorch program to train a network to high accuracy in a short time.

    Provide a concise two-paragraph summary that addresses the following points:

    Paragraph 1:
    - The number of epochs completed within the time limit and the speed of the model in terms of epochs per second
    - The mean accuracy achieved, standard deviation, training loss and other relevant metrics
    - The training behavior, including stability, convergence, overfitting, or underfitting
    - What this suggests about the programs hyper parameters, model size, or other factors

    Paragraph 2:
    - Evaluate the program's success in achieving the original plan's goals
    - Identify parts of the program that contributed to the observed behaviors
    - Suggest specific changes to improve the program's performance
    """)

    plan: str = dspy.InputField(desc="The intention for the program")
    program: str = dspy.InputField(desc="The program that was run")
    stdout: str = dspy.InputField(desc="The output of the program")
    summary: str = dspy.OutputField(desc="A concise two-paragraph summary addressing the specified points")


def evaluate(program, result):
    try:
        return dspy.TypedPredictor(EvalSignature)(
            plan=program.analysis,
            program=program.program,
            stdout=result["stdout"],
        ).summary
    except Exception as e:
        print(traceback.format_exc())
        print(f"Failed to explain program. Error: {e}")
        return
