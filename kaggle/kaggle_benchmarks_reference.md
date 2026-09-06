# kaggle-benchmarks Task Syntax Reference

- Installation: `pip install kaggle-benchmarks`
- [Quick Start](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/quick_start.md)
- [Cookbook](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md)

## Decorator & Signature

```python
@kbench.task(name="task_name", description="...", version=1)
def my_task(llm, param1: str, param2: int) -> ReturnType:
    ...
```

- First param is always `llm` (model under test).
- Additional params passed via `.run()` or `.evaluate()`.
- `name` defaults to function name; `description` defaults to docstring.
- **Important:** The `name` is normalized to a URL-safe slug (e.g. `"My Task"` becomes `my-task`).
  This slug must match the task name used with `kaggle b t push <task-slug> -f <file>`.

## Return Type Annotations (controls leaderboard rendering)

| Annotation | Meaning |
|---|---|
| `None` / omitted | Pass/Fail: graded solely by assertions |
| `-> bool` | Binary pass/fail |
| `-> int` / `-> float` | Numerical score |
| `-> tuple[int, int]` | (passed, total) count |
| `-> tuple[float, float]` | (value, confidence_interval) |
| `-> dict` | Structured result dict |

## LLM Interaction

```python
response = llm.prompt("question")                            # returns str
obj = llm.prompt("question", schema=MyDataclass)             # structured output
response = llm.prompt("q", image=images.from_url(url))       # with image
response = llm.prompt("q", video=videos.from_url(yt_url))    # with video
response = llm.prompt("q", audio=audios.from_path(path))     # with audio
response = llm.prompt("q", tools=[my_func])                  # tool calling (needs api="genai")
llm.send(msg)        # adds message without triggering response
llm.respond()         # gets response continuing existing conversation
kbench.user.send(x)   # adds user message (text/image/etc.) to chat history
```

## Chat Context Management

```python
with kbench.chats.new("name"):     # isolated chat context
    response = llm.prompt("...")    # only this chat's history is sent
```

Use `kbench.chats.new()` in loops to avoid growing context.
For multi-turn chats: `contexts.enter(chat=chat)`

## Accessing Models

```python
kbench.llm                              # default model placeholder
kbench.judge_llm                        # judge model
kbench.llms["google/gemini-2.5-flash"]  # specific model by name
```

## Assertions (always include `expectation=` for leaderboard display)

```python
kbench.assertions.assert_equal(expected, actual, expectation="...")
kbench.assertions.assert_true(value, expectation="...")
kbench.assertions.assert_false(value, expectation="...")
kbench.assertions.assert_in(member, container, expectation="...")
kbench.assertions.assert_not_in(member, container, expectation="...")
kbench.assertions.assert_contains_regex(pattern, text, expectation="...")
kbench.assertions.assert_not_contains_regex(pattern, text, expectation="...", flags=0)
kbench.assertions.assert_empty(container, expectation="...")
kbench.assertions.assert_not_empty(container, expectation="...")
kbench.assertions.assert_fail(expectation="...")  # unconditional fail
```

## Custom Assertions

```python
from kaggle_benchmarks.assertions import assertion_handler, AssertionResult

@assertion_handler()
def assert_is_positive(value: float, expectation: str) -> AssertionResult:
    return AssertionResult(passed=value > 0, expectation=expectation)
```

## Judge-Based Assessment

```python
report = kbench.assertions.assess_response_with_judge(
    criteria=("criterion 1", "criterion 2"),
    response_text=response,
    judge_llm=kbench.judge_llm,
    prompt_fn=optional_custom_fn,       # (criteria, response_text) -> str
    output_schema=OptionalDataclass,
)
for r in report.results:
    kbench.assertions.assert_true(r.passed, expectation=f"{r.criterion}: {r.reason}")
```

## Running Tasks

`.run()` or `.evaluate()` MUST be called to generate a run file.
Invoking one of these methods is required to generate the `.run.json` record and save evaluation results.

```python
# Single run:
run = my_task.run(llm=kbench.llm, param1="val1", param2=42)

# Dataset evaluation (runs task once per row in a DataFrame):
runs = my_task.evaluate(
    llm=[kbench.llm], evaluation_data=df,  # df columns map to task params
    n_jobs=2, timeout=120,
    stop_condition=lambda r: len(r) == df.shape[0],
    max_attempts=50, retry_delay=15, remove_run_files=True,
)
results_df = runs.as_dataframe()
```

## Multimodal Content Factories

```python
from kaggle_benchmarks.content_types import images, videos, audios
images.from_url(url) | images.from_path(p) | images.from_base64(b64, format="png")
videos.from_url(youtube_url)
audios.from_path(p) | audios.from_url(url) | audios.from_base64(b64, format="mp3")
```

## Token Usage Tracking

```python
with kbench.chats.new("chat") as chat:
    llm.prompt("...")
    chat.usage.input_tokens / .output_tokens / .input_tokens_cost_nanodollars
```
