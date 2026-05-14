# Answer Quality Evaluation Rubric

Use this rubric to manually compare the original fixed-strategy RAG and the new intent-aware RAG.

## Scoring Dimensions

Each answer is scored from 1 to 5 on the following dimensions:

### 1. Relevance

- 1: Mostly off-topic
- 2: Partly relevant but misses the user's main need
- 3: Generally relevant
- 4: Clearly relevant and focused
- 5: Fully aligned with the question intent

### 2. Groundedness

- 1: Mostly unsupported by retrieved material
- 2: Limited grounding
- 3: Partially grounded
- 4: Well grounded in retrieved content
- 5: Strongly grounded with clear use of course material

### 3. Completeness

- 1: Very incomplete
- 2: Misses major points
- 3: Covers core points only
- 4: Covers most important points
- 5: Complete for the intended task type

### 4. Structure and Clarity

- 1: Hard to follow
- 2: Weak structure
- 3: Understandable but plain
- 4: Clear and well structured
- 5: Very clear, organized, and easy to learn from

## Optional Binary Checks

Add a Yes/No note for:

- Hallucination risk
- Proper refusal or limitation statement when coverage is weak
- Fit with detected intent

## Suggested Summary Metrics

After scoring a batch, calculate:

- Average relevance
- Average groundedness
- Average completeness
- Average structure
- Average overall score
- Hallucination count

## How to Use This in Interviews

You can summarize the evaluation like this:

1. We first improved intent routing accuracy.
2. Then we manually compared answer quality across multiple dimensions.
3. We found that intent-aware prompting improved relevance, structure, and groundedness, especially for comparison, summary, and quiz-style questions.
