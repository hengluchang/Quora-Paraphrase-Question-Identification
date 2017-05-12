from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
UNI_BLEU_WEIGHTS = (1, 0, 0, 0)
BI_BLEU_WEIGHTS = (0, 1, 0, 0)
BLEU2_WEIGHTS = (0.5, 0.5, 0, 0)


def save_results(predictions,IDs,  filename):
    with open(filename, 'w') as f:
        f.write("test_id,is_duplicate\n")
        for i, pred in enumerate(predictions):
            f.write("%d,%f\n" % (IDs[i], pred))


"""Very basic tokenizer: split the sentence by space into a list of tokens."""
def tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
      words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


"""Character ngram tokenizer: split the sentence into a list of char ngram tokens."""
def char_ngram_tokenizer(sentence, n):
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]


def get_word_count(x):
    return len(tokenizer(str(x)))


def word_overlap(x):
    return len(set(str(x['question1']).lower().split()).intersection(
             set(str(x['question2']).lower().split())))


def char_bigram_overlap(x):
    return len(set(char_ngram_tokenizer(str(x['question1']), 2)).intersection(
             set(char_ngram_tokenizer(str(x['question2']), 2))))


def char_trigram_overlap(x):
    return len(set(char_ngram_tokenizer(str(x['question1']), 3)).intersection(
             set(char_ngram_tokenizer(str(x['question2']), 3))))


def char_fourgram_overlap(x):
    return len(set(char_ngram_tokenizer(str(x['question1']), 4)).intersection(
             set(char_ngram_tokenizer(str(x['question2']), 4))))


def get_uni_BLEU(x):
    s_function = SmoothingFunction()
    # method 2 is add 1 smoothing
    return sentence_bleu([tokenizer(str(x['question2']))],
                         tokenizer(str(x['question1'])),
                         weights=UNI_BLEU_WEIGHTS,
                         smoothing_function=s_function.method2)


def get_bi_BLEU(x):
    s_function = SmoothingFunction()
    # method 2 is add 1 smoothing
    return sentence_bleu([tokenizer(str(x['question2']))],
                         tokenizer(str(x['question1'])),
                         weights=BI_BLEU_WEIGHTS,
                         smoothing_function=s_function.method2)


def get_BLEU2(x):
    s_function = SmoothingFunction()
    # method 2 is add 1 smoothing
    return sentence_bleu([tokenizer(str(x['question2']))],
                         tokenizer(str(x['question1'])),
                         weights=BLEU2_WEIGHTS,
                         smoothing_function=s_function.method2)


def feature_eng(df):

    # word count of question 1
    df['q1_word_count'] = df['question1'].apply(get_word_count)

    # word count of question 2
    df['q2_word_count'] = df['question2'].apply(get_word_count)

    # word count difference
    df['word_count_diff'] = abs(df['q1_word_count'] - df['q2_word_count'])

    # number of word overlap between q1 and q2
    df['word_overlap'] = df.apply(word_overlap, axis=1)

    # unigram BLEU score
    df['uni_BLEU'] = df.apply(get_uni_BLEU, axis=1)

    # bigram BLEU score
    df['bi_BLEU'] = df.apply(get_bi_BLEU, axis=1)

    # BLEU2 score
    df['BLEU2'] = df.apply(get_BLEU2, axis=1)

    # character unigram overlap
    df['char_bigram_overlap'] = df.apply(char_bigram_overlap, axis=1)

    # character trigram overlap
    df['char_trigram_overlap'] = df.apply(char_trigram_overlap, axis=1)

    # character 4-gram overlap
    df['char_4gram_overlap'] = df.apply(char_fourgram_overlap, axis=1)

    return df