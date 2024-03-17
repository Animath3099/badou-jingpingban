from model import SiameseNetwork
from config import Config
import torch
import jieba
import json

def main(vec):
    def load_vocab(vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  #0留给padding位置，所以从1开始
        return token_dict

    target_to_questions = {}
    with open("../data/train.json", encoding="utf8") as f:
        for index, line in enumerate(f):
            content = json.loads(line)
            questions = content["questions"]
            target = content["target"]
            target_to_questions[target] = questions
    config = Config
    question = vec
    vocab = load_vocab("../chars.txt")
    config["vocab_size"] = len(vocab)
    model = SiameseNetwork(config)
    model.load_state_dict(torch.load("model_output/epoch_10.pth"))
    input_id = []
    for word in question:
        input_id.append(vocab.get(word, vocab["[UNK]"]))
    input_id = input_id[:config["max_length"]]
    input_id += [0] * (config["max_length"] - len(input_id))
    input_id = torch.LongTensor(input_id)
    model.eval()
    question = model(input_id)
    target_id = []
    target_name = []
    for target in target_to_questions.keys():
        a = []
        for word in target:
            a.append(vocab.get(word, vocab["[UNK]"]))
        a = a[:config["max_length"]]
        a += [0] * (config["max_length"] - len(a))
        target_name.append(target)
        target_id.append(a)
    target_num = len(target_id)
    target_id = torch.LongTensor(target_id)
    targets = model(target_id)

    question = torch.nn.functional.normalize(question.unsqueeze(0), dim=-1)
    question = question.repeat(target_num, 1)
    targets = torch.nn.functional.normalize(targets, dim=-1)
    score = torch.sum(torch.mul(question, targets), axis = -1).squeeze().tolist()
    result = []
    for i in range(target_num):
        result.append([target_name[i], score[i]])
    sort_results = sorted(result, key=lambda x:x[1], reverse=True)
    print(sort_results[:3])
    answer = []
    for ans in sort_results:
        print(ans)
        print("ans[i]:",ans[1])
        if ans[1] > 0.85:
            answer.append(ans)
    if len(answer) < 1:
        answer=[["对不起，没有查询到你所要咨询的业务",1.0]]
    return answer[0:6]
# vec = "我想重置下固话密码"
# main(vec)

while True:
    # 用户输入问题
    question_str = input("请输入问题（输入'quit'退出）: ")
    if question_str.lower() == 'quit':
        break

    # 获取答案
    answer = main(question_str)
    print(f"模型回答: {answer}")