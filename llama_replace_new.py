import re
import random
import subprocess
import string
import pandas as pd
import nltk

def create_prompt(list_of_paragraphs, list_of_picks):
    list_of_templates = []
    for i in range(len(list_of_paragraphs)):
        if i in list_of_picks:
            list_of_templates.append(cut_paragraph(list_of_paragraphs[i - 1]))
    return list_of_templates


def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )

def preprocess(article):
    # delete urls
    url_pattern = re.compile(
        r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*")
    article = re.sub(url_pattern, '', article)

    # reorder paragraphs
    paragraphs = article.split('\n')
    processed_paragraphs = []
    temp_paragraph = ""
    for number, paragraph in enumerate(paragraphs):
        # remove heading
        if number == 0:
            if len(nltk.tokenize.sent_tokenize(paragraph)) == 1:
                continue

        if paragraph.lower().startswith('image source') \
                or paragraph.lower().startswith('source') \
                or paragraph.lower().startswith('photo by'):
            continue
        elif paragraph == '':
            continue

        if len(paragraph) < 55:  # probabliy it is a subheading
            continue

        if len(paragraph) < 450 and temp_paragraph == "":  # the paragraph is too small
            temp_paragraph = paragraph
        elif temp_paragraph != "":
            temp_paragraph += paragraph
            temp_paragraph += " "
            if len(temp_paragraph) >= 500:
                processed_paragraphs.append(temp_paragraph)
                temp_paragraph = ""
        else:
            processed_paragraphs.append(paragraph)

    return processed_paragraphs


def pick_paragraphs_to_be_replaced(paragraphs_cnt: int, number_to_pick: int):
    l = range(1, paragraphs_cnt)
    try:
        k = random.sample(l, number_to_pick)
    except ValueError as e:
        return []
    return sorted(k)


def postprocess(llama_ans):
    paragraphs = llama_ans.split('\n')
    llama_ans = llama_ans.replace('\r\n', ' ')
    postprocessed = ""
    if 400 <= len(paragraphs[0]) <= 600:
        return paragraphs[0]
    for paragraph in paragraphs:
        paragraph = remove_non_ascii(paragraph)
        postprocessed += paragraph
        if len(postprocessed) >= 600:
            postprocessed = cut_paragraph(postprocessed)
            return postprocessed
    return cut_paragraph(llama_ans)

punkt = re.compile('([.?!]\s*)')
def cut_paragraph(paragraph):
    sentences = re.split(punkt, paragraph)
    cur_paragraph = ""
    len_of_cur_paragraph = 0
    for idx in range(len(sentences)):
        sent = sentences[idx]
        if sent == "" or re.match(punkt, sent):
            continue
        l = len(sent)
        if len_of_cur_paragraph + l > 700:
            break
        else:
            cur_paragraph += sent
            if idx + 1 < len(sentences) and re.match(punkt, sentences[idx + 1]):
                punkt_sign = sentences[idx + 1].strip()
                cur_paragraph += punkt_sign + " "
            len_of_cur_paragraph += l + 2
    return cur_paragraph


terminal_template = "./main -m ./models/7B/ggml-model-q4_0.bin -c 1000 --simple-io --repeat-penalty 1.5 --n-predict 1000 -p"


def call_llama(prompt_templates):
    llama_answers = []
    for prompt in prompt_templates:
        prompt_len = len(prompt)
        # returns output as byte string
        cmd = f'{terminal_template}  "{prompt}"'
        returned_output = ""

        # using decode() function to convert byte string to string
        idx = 0
        while returned_output == "" and idx < 5:
            try:
                returned_output = subprocess.check_output(cmd)
            except subprocess.CalledProcessError as e:
                print(e.output)
                idx += 1
                continue
        if returned_output == "":
            return []
        llama_answer = returned_output.decode('utf-8', errors='ignore')
        length = len(llama_answer) - prompt_len
        while "SEGV" in llama_answer or length < 400:
            try:
                returned_output = subprocess.check_output(cmd)
            except subprocess.CalledProcessError as e:
                print(e.output)
                continue
            llama_answer = returned_output.decode('utf-8', errors='ignore')
            length = len(llama_answer) - len(prompt)
        length = len(llama_answer) - prompt_len
        llama_answers.append(llama_answer[prompt_len - 1:])

    return llama_answers


def check_for_correctness(paragraph, number):
    if paragraph == "":
        with open(r"documents_new\ailed.txt", 'a') as file2:
            file2.write(str(number))
            file2.write("\n")
        return False
    return True


def process_article(article, article_num, number_to_replace):
    list_of_paragraphs = preprocess(article)
    list_of_picks = pick_paragraphs_to_be_replaced(len(list_of_paragraphs), number_to_replace)
    if (list_of_picks == []):
        check_for_correctness("", article_num)
        return
    prompt_templates = create_prompt(list_of_paragraphs, list_of_picks)
    llama_answers = call_llama(prompt_templates)
    if llama_answers == []:
        check_for_correctness("", article_num)
        return
    idx = 0
    with open(f"documents_new\picks_{article_num}_1.txt", 'w', encoding='utf-8') as file:
        for i in list_of_picks:
            file.write(f"{i} ")
    with open(f"documents_new\output_{article_num}_1.txt", 'w', encoding='utf-8') as file:
        for j in range(len(list_of_paragraphs)):
            if j in list_of_picks:
                text = postprocess(llama_answers[idx])
                idx += 1
            else:
                text = postprocess(list_of_paragraphs[j])
            file.write(text)
            file.write("\n")

        # for i in range(len(prompt_templates)):

# nltk.download('punkt')
#
dataset = pd.read_csv('medium.csv')


# for article, number in enumerate(dataset[0][210:1328]):
#     process_article(article, number, 2)
# for article, number in enumerate(dataset[0][1328:2656]):
#     process_article(article, number, 3)
# for article, number in enumerate(dataset[0][2656:]):
#     process_article(article, number, 4)

