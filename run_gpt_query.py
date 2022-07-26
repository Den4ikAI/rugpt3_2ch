"""
Пример инференса модели https://huggingface.co/Den4ikAI/rugpt3_2ch
"""

import math
import os.path

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   

    tokenizer = AutoTokenizer.from_pretrained("Den4ikAI/rugpt3_2ch")
    tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'pad_token': '<pad>'})

    model = AutoModelForCausalLM.from_pretrained("Den4ikAI/rugpt3_2ch")
    model.to(device)
    model.eval()

    while 1:
        test_in = input("~| ")
        prompt = "- {}\n-".format(test_in)

        encoded_prompt = tokenizer.encode(prompt, return_tensors="pt").to(device)
        out = model.generate(encoded_prompt, max_length=200, do_sample=True, top_k=35, top_p=0.85, temperature=1.0,
                            num_return_sequences=10, eos_token_id=2, pad_token_id=0)

        
        for i, tokens in enumerate(out.cpu().tolist(), start=1):
            tokens = tokens[encoded_prompt.shape[1]:]
            text = tokenizer.decode(tokens)
            reply = text[:text.index('\n')]
            print('[{}] - {}'.format(i, reply))
