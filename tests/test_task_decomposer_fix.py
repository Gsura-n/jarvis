from jarvis import task_decomposer


def test_synthesis_failure_keeps_previous_results(monkeypatch):
    state = {
        "prompt": "Test Prompt",
        "results": [
            {"id": 1, "task": "Task 1", "intent": "general", "result": "a" * 1600},
            {"id": 2, "task": "Task 2", "intent": "general", "result": "b" * 1600},
            {"id": 3, "task": "Task 3", "intent": "general", "result": "c" * 1600},
            {"id": 4, "task": "Task 4", "intent": "general", "result": "d" * 1600},
        ],
    }

    # Mock smart_compress to pass through strings directly
    monkeypatch.setattr(task_decomposer, "smart_compress", lambda r: r["result"])

    call_count = 0

    def mock_call_model(model, system, user, *args, **kwargs):
        nonlocal call_count
        if "technical writer" in system:
            call_count += 1
            if call_count == 2:  # Fail on step 2
                return "[ERROR] failed synthesis"
            return user
        return "compressed" * 5

    monkeypatch.setattr(task_decomposer, "call_model", mock_call_model)

    new_state = task_decomposer.aggregate_results(state)
    ans = new_state["final_answer"]
    assert "a" * 1600 in ans
    assert "b" * 1600 in ans
    assert "[ERROR]" not in ans


def test_synthesis_all_fail(monkeypatch):
    state = {
        "prompt": "Test Prompt",
        "results": [
            {"id": 1, "task": "Task 1", "intent": "general", "result": "a" * 1600},
            {"id": 2, "task": "Task 2", "intent": "general", "result": "b" * 1600},
            {"id": 3, "task": "Task 3", "intent": "general", "result": "c" * 1600},
            {"id": 4, "task": "Task 4", "intent": "general", "result": "d" * 1600},
        ],
    }

    # Mock smart_compress to pass through strings directly
    monkeypatch.setattr(task_decomposer, "smart_compress", lambda r: r["result"])

    def mock_call_model(model, system, user, *args, **kwargs):
        if "technical writer" in system:
            return "[ERROR] failed synthesis completely"
        return "compressed" * 5

    monkeypatch.setattr(task_decomposer, "call_model", mock_call_model)

    new_state = task_decomposer.aggregate_results(state)
    ans = new_state["final_answer"]
    assert "a" * 1600 in ans
    assert "b" * 1600 in ans
    assert "c" * 1600 in ans
    assert "d" * 1600 in ans


def test_smart_compress_reasoning_fails(monkeypatch):
    result = {"id": 1, "task": "Task 1", "intent": "reasoning", "result": "x" * 2000}

    def mock_call_model(model, system, user, *args, **kwargs):
        return "[ERROR] compression failed"

    monkeypatch.setattr(task_decomposer, "call_model", mock_call_model)
    compressed = task_decomposer.smart_compress(result)
    assert compressed == "x" * 2000


def test_smart_compress_code_fails(monkeypatch):
    code_block = "```python\nprint(1)\n```"
    prose = "long prose " * 100
    text = prose + code_block + prose + code_block + prose
    result = {"id": 1, "task": "Task 1", "intent": "code", "result": text}

    def mock_call_model(model, system, user, *args, **kwargs):
        return "[ERROR] compression failed"

    monkeypatch.setattr(task_decomposer, "call_model", mock_call_model)
    compressed = task_decomposer.smart_compress(result)

    assert code_block in compressed
    assert compressed.count("```python") == 2


def test_happy_path(monkeypatch):
    state = {
        "prompt": "Test Prompt",
        "results": [
            {"id": 1, "task": "Task 1", "intent": "general", "result": "a" * 1600},
            {"id": 2, "task": "Task 2", "intent": "general", "result": "b" * 1600},
            {"id": 3, "task": "Task 3", "intent": "general", "result": "c" * 1600},
            {"id": 4, "task": "Task 4", "intent": "general", "result": "d" * 1600},
        ],
    }

    def mock_call_model(model, system, user, *args, **kwargs):
        if "technical writer" in system:
            return f"Synthesized: {user}"
        return "compressed" * 5

    monkeypatch.setattr(task_decomposer, "call_model", mock_call_model)
    new_state = task_decomposer.aggregate_results(state)
    ans = new_state["final_answer"]
    assert "Synthesized:" in ans
    assert (
        ans
        == "Synthesized: Running synthesis so far:\nSynthesized: Running synthesis so far:\nSynthesized: Running synthesis so far:\n# Test Prompt\n\n## Task 1\n\n"
        + "compressed" * 5
        + "\n\n\nNew subtask result to integrate:\n## Task 2\n"
        + "compressed" * 5
        + "\n\nNew subtask result to integrate:\n## Task 3\n"
        + "compressed" * 5
        + "\n\nNew subtask result to integrate:\n## Task 4\n"
        + "compressed" * 5
    )
    assert "[ERROR]" not in ans
