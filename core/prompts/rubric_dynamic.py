
RUBRIC_GENERATION_PROMPT = (
    "As an expert educator, your task is to craft a detailed rubric for evaluating how student's code submission compares to a reference code. The rubric should be tailored to the reference code.\n\n"
    "Reference Code:\n"
    "--------------------\n"
    "{code}\n"
    "--------------------\n\n"
    "Summarize the code in a few sentences."
    "Evaluate student code submissions based on their semantic similarity to the reference code. "
    "Assess how well the student code captures the same functionality, logic, and intent as the reference code, "
    "regardless of differences in implementation details or coding style.\n\n"
)
