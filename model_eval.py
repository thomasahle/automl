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


# FIXME: The model sometimes return kinda annoying summaries like this:
#
# Plan: The goal was to train a neural network to achieve high accuracy in a short amount of time without a reference to previous models for comparison.
#
# Program: The provided PyTorch program defines a convolutional neural network for image classification, including two convolutional layers followed by max pooling, and two fully connected layers. It uses the Adam optimizer with a learning rate scheduler and cross-entropy loss function.
#
# Stdout: The output shows the program was run three times, each with a different number of epochs completed within a 5-second time limit. The first run completed 5.9 epochs, the second 6.05, and the third 6.02. The speed of the model varied slightly across runs but generally completed around 1.2 epochs per second. The training loss decreased and test accuracy increased over epochs in all runs, indicating stable training and convergence. The final accuracy reached was around 63%, with a slight improvement over epochs but no clear signs of overfitting or underfitting within the limited epoch range.
#
# Summary: The program successfully trained a neural network to a reasonable accuracy within a very short time frame, achieving the goal of high accuracy in a limited time. The training was stable across runs, with consistent improvements in test accuracy and reductions in training loss, indicating effective convergence. There was no clear evidence of overfitting or underfitting, likely due to the limited number of epochs run. The use of the Adam optimizer and a learning rate scheduler likely contributed to the stable and effective training behavior observed.
#
# To further improve these behaviors, modifications could include experimenting with deeper or more complex network architectures to enhance learning capacity, adjusting the learning rate or scheduler parameters for potentially faster convergence, and incorporating regularization techniques or data augmentation to combat overfitting if training for more epochs. Additionally, increasing the batch size, if memory permits, could improve the stability and efficiency of training. These changes could help in achieving higher accuracy or maintaining good performance over a longer training period.
