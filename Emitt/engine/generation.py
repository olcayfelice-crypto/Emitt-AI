import os
import sys
import numpy as np


CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    CURRENT_DIR
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

if CURRENT_DIR not in sys.path:
    sys.path.insert(
        0,
        CURRENT_DIR
    )


from tokenizer import EmittTokenizer
from model import EmittModel


class EmittGenerator:

    def __init__(
        self,
        vocabulary_path,
        model_path,
        max_tokens=20,
        min_tokens=1
    ):

        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

        print("Tokenizer yükleniyor...")

        self.tokenizer = EmittTokenizer()

        self.tokenizer.load(
            vocabulary_path
        )

        print("Model yükleniyor...")

        self.model = EmittModel.load(
            model_path
        )

        print("Emitt hazır.")

    def select_next_token(
        self,
        token_ids,
        generated_count
    ):

        probabilities = self.model.probabilities(
            token_ids
        )

        pad_id = self.tokenizer.token_to_id[
            "<PAD>"
        ]

        bos_id = self.tokenizer.token_to_id[
            "<BOS>"
        ]

        eos_id = self.tokenizer.token_to_id[
            "<EOS>"
        ]

        unk_id = self.tokenizer.token_to_id[
            "<UNK>"
        ]

        candidates = []

        for token_id, probability in enumerate(
            probabilities
        ):

            if token_id == pad_id:
                continue

            if token_id == bos_id:
                continue

            if token_id == unk_id:
                continue

            if (
                token_id == eos_id
                and generated_count < self.min_tokens
            ):
                continue

            candidates.append(
                (
                    float(probability),
                    token_id
                )
            )

        if not candidates:
            return eos_id

        candidates.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return candidates[0][1]

    def generate(
        self,
        prompt
    ):

        if not prompt.strip():
            return ""

        prompt_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False
        )

        bos_id = self.tokenizer.token_to_id[
            "<BOS>"
        ]

        eos_id = self.tokenizer.token_to_id[
            "<EOS>"
        ]

        token_ids = [
            bos_id
        ]

        token_ids.extend(
            prompt_ids
        )

        generated_ids = []

        for generated_count in range(
            self.max_tokens
        ):

            next_token = self.select_next_token(
                token_ids,
                generated_count
            )

            if next_token == eos_id:
                break

            token_ids.append(
                next_token
            )

            generated_ids.append(
                next_token
            )

        if not generated_ids:
            return ""

        response = self.tokenizer.decode(
            generated_ids,
            remove_special_tokens=True
        )

        return response.strip()


def main():

    print()
    print("=================================")
    print("       EMITT ")
    print("=================================")
    print()

    vocabulary_path = os.path.join(
        DATA_DIR,
        "vocabulary.json"
    )

    model_path = os.path.join(
        DATA_DIR,
        "emitt_model.json"
    )

    if not os.path.exists(
        vocabulary_path
    ):

        print(
            "HATA: vocabulary.json bulunamadı."
        )

        print(
            vocabulary_path
        )

        return

    if not os.path.exists(
        model_path
    ):

        print(
            "HATA: emitt_model.json bulunamadı."
        )

        print(
            model_path
        )

        return

    try:

        generator = EmittGenerator(
            vocabulary_path=
                vocabulary_path,

            model_path=
                model_path,

            max_tokens=20,

            min_tokens=1
        )

    except Exception as error:

        print()
        print(
            "MODEL YÜKLEME HATASI:"
        )

        print(
            error
        )

        return

    print()
    print("---------------------------------")
    print("Emitt kullanıma hazır.")
    print("Çıkmak için: exit")
    print("---------------------------------")
    print()

    while True:

        try:

            prompt = input(
                "Sen: "
            )

            if prompt.strip().lower() == "exit":

                print()
                print(
                    "Emitt kapatılıyor."
                )

                break

            if not prompt.strip():
                continue

            response = generator.generate(
                prompt
            )

            print(
                "Emitt:",
                response
            )

            print()

        except KeyboardInterrupt:

            print()
            print(
                "Emitt kapatılıyor."
            )

            break

        except Exception as error:

            print()
            print(
                "ÜRETİM HATASI:"
            )

            print(
                error
            )

            print()


if __name__ == "__main__":

    main()