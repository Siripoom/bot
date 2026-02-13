from scripts.benchmark_chatbot_vs_gemini import compute_binary_metrics


def test_compute_binary_metrics_basic() -> None:
    rows = [
        {"chatbot_is_correct": True, "gemini_is_correct": False},
        {"chatbot_is_correct": False, "gemini_is_correct": False},
        {"chatbot_is_correct": True, "gemini_is_correct": True},
    ]
    chatbot = compute_binary_metrics(rows, prefix="chatbot")
    gemini = compute_binary_metrics(rows, prefix="gemini")

    assert chatbot["correct_count"] == 2
    assert chatbot["incorrect_count"] == 1
    assert chatbot["correct_pct"] == 66.67
    assert chatbot["incorrect_pct"] == 33.33

    assert gemini["correct_count"] == 1
    assert gemini["incorrect_count"] == 2
    assert gemini["correct_pct"] == 33.33
    assert gemini["incorrect_pct"] == 66.67


def test_compute_binary_metrics_zero_rows() -> None:
    result = compute_binary_metrics([], prefix="chatbot")
    assert result["correct_count"] == 0
    assert result["incorrect_count"] == 0
    assert result["correct_pct"] == 0.0
    assert result["incorrect_pct"] == 0.0
