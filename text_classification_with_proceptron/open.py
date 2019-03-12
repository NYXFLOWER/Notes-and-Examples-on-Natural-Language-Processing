import re, time


if __name__ == '__main__':
    corpus_path = '/Users/nyxfer/Documents/GitHub/nlp/sentence_completion/news-corpus-500k.txt'
    question_path = '/Users/nyxfer/Documents/GitHub/nlp/sentence_completion/questions.txt'

    t = time.time()
    with open(corpus_path, 'r') as f:
        s = f.read().lower()
    s = re.sub(r'\n', 'pkpkllll', s)
    s = re.sub("[^\w]", " ", s)
    s = re.sub(r'pkpkllll', '</s> <s> ', s).split()
    print(time.time() - t)

    t = time.time()
    with open(corpus_path, 'r') as f:
        s = f.readlines()

    a = []
    for line in s:
        a.append("<s>")
        a.extend(re.sub("[^\w]", '', line.lower()).split())
        a.append('</s>')
    print(time.time() - t)

    t = time.time()
    with open(corpus_path, 'r') as f:
        s = f.readlines()
    s = [' '.join(['<s>', re.sub("[^\w]", " ", line), '</s>']) for line in s]
    s = ' '.join(s)
    s = re.split(" ", s)
    print(time.time() - t)
