import re
import json
from collections import Counter


class EmittTokenizer:
    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ]

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}

        self._create_special_tokens()

    def _create_special_tokens(self):
        for token in self.SPECIAL_TOKENS:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.token_to_id:
            token_id = len(self.token_to_id)

            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    def split_text(self, text):
        """
        Metni kelime, sayı ve noktalama parçalarına ayırır.
        """

        text = text.strip()

        if not text:
            return []

        return re.findall(
            r"\w+|[^\w\s]",
            text,
            flags=re.UNICODE
        )

    def build_vocabulary(self, texts, min_frequency=1):
        """
        Eğitim metinlerinden Emitt'in vocabulary'sini oluşturur.
        """

        counter = Counter()

        for text in texts:
            tokens = self.split_text(text)

            for token in tokens:
                counter[token] += 1

        for token, frequency in counter.items():
            if frequency >= min_frequency:
                self._add_token(token)

    def encode(self, text, add_special_tokens=True):
        """
        Yazıyı token ID'lerine dönüştürür.
        """

        tokens = self.split_text(text)

        ids = []

        if add_special_tokens:
            ids.append(self.token_to_id["<BOS>"])

        for token in tokens:
            token_id = self.token_to_id.get(
                token,
                self.token_to_id["<UNK>"]
            )

            ids.append(token_id)

        if add_special_tokens:
            ids.append(self.token_to_id["<EOS>"])

        return ids

    def decode(self, ids, remove_special_tokens=True):
        """
        Token ID'lerini tekrar yazıya dönüştürür.
        """

        tokens = []

        for token_id in ids:
            token = self.id_to_token.get(
                int(token_id),
                "<UNK>"
            )

            if remove_special_tokens:
                if token in self.SPECIAL_TOKENS:
                    continue

            tokens.append(token)

        text = ""

        for token in tokens:

            if not text:
                text = token

            elif re.match(r"[.,!?;:%)\]}]", token):
                text += token

            elif token in ["(", "[", "{"]:
                text += " " + token

            else:
                text += " " + token

        return text

    def save(self, path):
        """
        Vocabulary'yi dosyaya kaydeder.
        """

        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": {
                str(k): v
                for k, v in self.id_to_token.items()
            }
        }

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2
            )

    def load(self, path):
        """
        Daha önce oluşturulmuş vocabulary'yi yükler.
        """

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        self.token_to_id = {
            str(k): int(v)
            for k, v in data["token_to_id"].items()
        }

        self.id_to_token = {
            int(k): v
            for k, v in data["id_to_token"].items()
        }

    @property
    def vocabulary_size(self):
        return len(self.token_to_id)


if __name__ == "__main__":

    tokenizer = EmittTokenizer()

    training_texts = [
        "Merhaba Emitt.",
        "Ben Emitt yapay zekasıyım.",
        "Nasılsın?",
        "Ben iyiyim.",
        "Bugün hava çok güzel.",
        "Yapay zeka öğreniyorum.",
    ]

    tokenizer.build_vocabulary(training_texts)

    text = "Merhaba Emitt!"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("Emitt Tokenizer")
    print("----------------")
    print("Metin      :", text)
    print("Token ID   :", encoded)
    print("Geri dönüş :", decoded)
    print("Vocabulary :", tokenizer.vocabulary_size)

    tokenizer.save("vocabulary.json")

    print()
    print("Vocabulary kaydedildi: vocabulary.json")