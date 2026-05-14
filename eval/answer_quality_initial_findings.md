# Initial Answer Quality Findings

This note records the first manual findings from the generated side-by-side review draft.

## Scope

Current draft coverage:

- Case 1: definition
- Case 2: comparison

The full workflow is ready for more cases, but these two are already enough to extract early conclusions.

## Case 1: "What is classification?"

### Baseline behavior

- The answer is detailed and strongly grounded.
- However, it is too long for a definition-style question.
- It reads more like a full explanation note than a clean course-assistant response.

### Intent-aware behavior

- The answer is better aligned with the `definition` intent.
- It starts with a short definition, then gives intuition, key points, and an example.
- Structure is cleaner and more suitable for teaching or demo presentation.

### Manual judgment

- Winner: `intent_aware_rag`
- Why:
  - Better intent fit
  - Better structure
  - More concise while still grounded

## Case 2: "classification 和 regression 有什么区别？"

### Baseline behavior

- The baseline answer provides a lot of content and some structured comparison.
- It is still influenced by the generic QA strategy and becomes long and heavy.
- It does not consistently frame the response as an explicit comparison-first answer.

### Intent-aware behavior

- The answer clearly switches into comparison mode.
- It separates concept explanations and comparison structure more explicitly.
- It is better aligned with how a user expects a "difference / compare" question to be answered.

### Manual judgment

- Winner: `intent_aware_rag`
- Why:
  - Clearer comparison structure
  - Better alignment with the expected task
  - More explainable as an intent-routed improvement

## Early Conclusion

Even from only the first two cases, the new intent-aware design already shows a meaningful quality gain:

- `definition` questions become cleaner and less overloaded
- `comparison` questions become more structured and task-aligned

This is exactly the kind of improvement that is easy to explain in interviews:

1. The baseline system could answer, but often used one generic response style.
2. The new system first classifies intent.
3. Then it changes retrieval strategy and prompt style accordingly.
4. As a result, answer format becomes more appropriate for the user's task.
