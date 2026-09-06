# Syntax reference: kaggle_benchmarks_reference.md
import kaggle_benchmarks as kbench

@kbench.task(name="What is Kaggle?", description="Does the LLM know what Kaggle is?")
def what_is_kaggle(llm) -> None:
    response = llm.prompt("What is Kaggle?")
    kbench.assertions.assert_in("platform", response.lower())

what_is_kaggle.run(kbench.llm)
