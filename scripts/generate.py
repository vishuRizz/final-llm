import argparse

from src.mini_llm.infer import generate_reply
from src.mini_llm.runtime import pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with mini LLM")
    parser.add_argument("--prompt", type=str, default="hi", help="Prompt string")
    parser.add_argument("--tokens", type=int, default=80, help="Number of new tokens")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling (0 disables)")
    args = parser.parse_args()

    print(f"Using device: {pick_device()}")
    print(f"Prompt: {repr(args.prompt)}")
    print("\n=== Generated Text ===\n")
    print(
        generate_reply(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
    )


if __name__ == "__main__":
    main()
