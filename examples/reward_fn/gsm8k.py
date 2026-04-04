import re

def extract_final_answer(text: str) -> str | None:
    match = re.search(r"####\s*([\-0-9\.,]+)", text)
    if match is None:
        return None
    return match.group(1).replace(",", "").strip()

def compute_reward(prompt, response, sample):
    predicted = extract_final_answer(response)
    target = str(sample.get("expected_response") or sample["reward_model"]["ground_truth"]).strip()
    if predicted is None:
        return {"score": 0.0, "format_ok": 0.0}
    return {
        "score": 1.0 if predicted == target else 0.0,
        "format_ok": 1.0,
        "predicted_answer": predicted,
    }
