from nltk.tokenize import sent_tokenize
import regex as re


def clean_text(text: str) -> str:
    sentences = sent_tokenize(text)

    # Remove newlines within sentences
    cleaned_sentences = [sentence.replace("\n", " ") for sentence in sentences]

    def clean_sent(sentence):
        # Remove extra spaces
        sentence = re.sub(r"\s+", " ", sentence)
        # Remove leading and trailing spaces
        sentence = sentence.strip()
        # Remove spaces after ' and before ".,-’"
        sentence = re.sub(r"’ ", "", sentence)
        sentence = re.sub(r" ’", "", sentence)
        sentence = re.sub(r" \.", "", sentence)
        sentence = re.sub(r" \,", "", sentence)
        sentence = re.sub(r" \-", "", sentence)

        return sentence

    # for each sentence, clean it and join together
    cleaned_text = "\n".join([clean_sent(sentence) for sentence in cleaned_sentences])
    return cleaned_text
