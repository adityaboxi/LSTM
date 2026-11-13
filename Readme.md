# LSTM Next-Word Prediction (Text Generation)

A word-level language model built with **TensorFlow / Keras**. It learns patterns from a text corpus (CampusX DSMP FAQs + beginner ML/DL notes) and **generates text one word at a time** given a starting prompt.

---

## 📌 Project Goal

Train an LSTM to predict the **next word** given the previous words.

**Example:**
* **Seed:** `"what is the fee"`
* **Output:** `"what is the fee for the course"`

This follows the same core principle behind autocomplete and large language models, implemented at a smaller scale.

---

## 🧠 The 4 Steps of the Code

The entire `index.py` is organized into four clearly separated sections. Memorize these four steps — they form the core pipeline of any LSTM text-generation project:

### ✅ Step 1 — Tokenization
Convert raw text into integer tokens and compute the total vocabulary size.

```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts([faqs])
vocab_size = len(tokenizer.word_index) + 1