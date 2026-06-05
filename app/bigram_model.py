class BigramModel:
    def __init__(self, text):
        self.bigrams = {}
        self.build_bigrams(text)

    def build_bigrams(self, text):
        sentences = text if isinstance(text, list) else [text] 
        for sentence in sentences:
            words = sentence.split()
            for i in range(len(words) - 1):
                bigram = (words[i], words[i + 1])
                if bigram in self.bigrams:
                    self.bigrams[bigram] += 1
                else:
                    self.bigrams[bigram] = 1

    def get_bigram_count(self, word1, word2):
        return self.bigrams.get((word1, word2), 0)

    def get_most_common_bigrams(self, n=10):
        return sorted(self.bigrams.items(), key=lambda item: item[1], reverse=True)[:n]
    
    def generate_text(self, start_word, length):
        current_word = start_word
        generated_text = [current_word]

        for _ in range(length - 1):
            next_words = [(bigram[1], count) for bigram, count in self.bigrams.items() if bigram[0] == current_word]
            if not next_words:
                break
            next_word = max(next_words, key=lambda x: x[1])[0]
            generated_text.append(next_word)
            current_word = next_word

        return ' '.join(generated_text)