import torch

from cs336_basics.decoding import apply_top_p, generate_completion, generate_token_ids, sample_next_token


class ScriptedLM(torch.nn.Module):
    context_length = 4

    def __init__(self, next_token_ids: list[int], vocab_size: int) -> None:
        super().__init__()
        self.next_token_ids = next_token_ids
        self.vocab_size = vocab_size
        self.calls = 0

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        next_token_id = self.next_token_ids[min(self.calls, len(self.next_token_ids) - 1)]
        self.calls += 1

        logits = torch.full(
            (token_ids.shape[0], token_ids.shape[1], self.vocab_size),
            -100.0,
            device=token_ids.device,
        )
        logits[:, -1, next_token_id] = 100.0
        return logits


class TinyTokenizer:
    text_to_id = {"a": 0, "b": 1, "<|endoftext|>": 2}
    id_to_text = {0: "a", 1: "b", 2: "<|endoftext|>"}

    def encode(self, text: str) -> list[int]:
        return [self.text_to_id[text]]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_text[token_id] for token_id in ids)


def test_temperature_zero_is_greedy():
    logits = torch.tensor([1.0, 3.0, 2.0])
    assert sample_next_token(logits, temperature=0).item() == 1


def test_top_p_keeps_smallest_prefix_reaching_threshold():
    probs = torch.tensor([0.5, 0.3, 0.2])
    filtered = apply_top_p(probs, top_p=0.6)
    expected = torch.tensor([0.625, 0.375, 0.0])
    torch.testing.assert_close(filtered, expected)


def test_generate_token_ids_stops_at_end_token():
    model = ScriptedLM(next_token_ids=[1, 2, 0], vocab_size=3)
    output = generate_token_ids(
        model=model,
        prompt_token_ids=[0],
        max_new_tokens=10,
        end_token_id=2,
        temperature=0,
    )
    assert output == [1]
    assert model.calls == 2


def test_generate_completion_decodes_text_without_end_token():
    model = ScriptedLM(next_token_ids=[1, 2], vocab_size=3)
    tokenizer = TinyTokenizer()
    completion = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt="a",
        max_new_tokens=10,
        temperature=0,
    )
    assert completion == "b"
