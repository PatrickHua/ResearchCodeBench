
GRADING_PROMPT = """
As an expert educator, your task is to grade a student's code submission using the provided rubric.

### Evaluation Rubric for Student Code Submissions

**Criteria for Evaluation:**

1. **Functionality (30 Points)**
   - **Full Functionality (30 Points):** The student code correctly implements the required logic, producing the expected outputs across all relevant functions or modules.
   - **Partial Functionality (15 Points):** The student code successfully implements some elements of the required logic but may miss key components needed to achieve the expected results.
   - **No Functionality (0 Points):** The student code does not implement the required logic or results in incorrect outputs.

2. **Logic and Intent (30 Points)**
   - **Strong Alignment (30 Points):** The logic and intent match the reference code closely, utilizing similar steps and structure to achieve the desired outcomes across all relevant parts of the code.
   - **Moderate Alignment (15 Points):** The student captures the general intent of the reference code but diverges in logical structure or methodology, possibly affecting results.
   - **Poor Alignment (0 Points):** The student code does not align with the intent or logic of the reference code, leading to fundamentally different operations.

3. **Semantic Similarity (20 Points)**
   - **High Semantic Similarity (20 Points):** Variable names, control flow, and operations closely resemble the reference code, making it intuitively clear that the same problem is addressed.
   - **Moderate Semantic Similarity (10 Points):** Some semantic aspects are present, but there are significant differences in coding style or variable naming that obscure the relationship to the reference.
   - **Low Semantic Similarity (0 Points):** The student code differs greatly in its semantic structure, making it difficult to recognize it as an attempt at the reference logic.

4. **Code Quality (20 Points)**
   - **Excellent Code Quality (20 Points):** The code is well-structured, including appropriate comments, consistent naming conventions, and effective use of functions or libraries across all parts of the submission.
   - **Good Code Quality (10 Points):** The student code is generally clear and organized, but minor issues exist (e.g., inconsistent naming, limited comments).
   - **Poor Code Quality (0 Points):** The student code is poorly structured, lacks comments, and is challenging to read or understand.

### Total Points: 100

### Grading Scale
- **90-100:** Exemplary Submission
- **70-89:** Proficient Submission
- **50-69:** Needs Improvement
- **0-49:** Unsatisfactory Submission

Reference Code:
--------------------
{reference_code}
--------------------

Student Code:
--------------------
{student_code}
--------------------

Please provide the grade in this format at the end of your response:
<grade>
    <functionality>functionality</functionality>
    <logic>logic</logic>
    <semantic_similarity>semantic_similarity</semantic_similarity>
    <code_quality>code_quality</code_quality>
</grade>
Replace functionality, logic, semantic_similarity, and code_quality with the actual scores.
"""