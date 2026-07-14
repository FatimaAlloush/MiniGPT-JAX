from inference.generate import generate_story
from inference.loader import load_model
import configs


def main():

    pretrained_model = load_model(
        configs.PRETRAINED_CHECKPOINT
    )

    trained_model = load_model(
        configs.TRAINED_CHECKPOINT
    )


    trained_output = generate_story(
        trained_model,
        configs.PROMPT,
        temperature=configs.TEMPERATURE,
        max_new_tokens=configs.MAX_NEW_TOKENS
    )


    pretrained_output = generate_story(
        pretrained_model,
        configs.PROMPT,
        temperature=configs.TEMPERATURE,
        max_new_tokens=configs.MAX_NEW_TOKENS
    )


    print("=" * 60)
    print("Trained MiniGPT")
    print("=" * 60)
    print(trained_output)


    print("\n")


    print("=" * 60)
    print("Pretrained MiniGPT")
    print("=" * 60)
    print(pretrained_output)


if __name__ == "__main__":
    main()