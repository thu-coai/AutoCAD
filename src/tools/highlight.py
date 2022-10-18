import json
from tqdm import tqdm
import os
from transformers import AutoTokenizer
import numpy as np
import argparse
from copy import deepcopy


LABELS = ['entailment', 'contradiction', 'neutral']


def convert_saliency_to_highlight(args):
    for split in ['train', 'dev']:
        with open(args.input_data_prefix + f'{split}.json', 'r') as f:
            data = json.load(f)
        
        with open(args.input_data_prefix + f'{split}_saliency.json', 'r') as f:
            saliency_data = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(args.nlu_model)
        
        for item, saliency_item in tqdm(zip(data, saliency_data)):
            if saliency_item['label'] == saliency_item['pred'] and saliency_item['prob'][saliency_item['label']] > args.min_confidence:
                label, premise, hypothesis = item['label'], item['premise'], item['hypothesis']
                seq_ids = tokenizer(premise, hypothesis)['input_ids']
                tokens = tokenizer.convert_ids_to_tokens(seq_ids)
                tokens = [i.strip('Ġ') for i in tokens] # clean up 
                ref_tokens = premise.split() + hypothesis.split() # reference: 按空格分隔

                translate_saliency = []
                translate_tokens = []
                tmp_subword_saliency_list = []
                tmp_subword = ''

                ref_id = 0
                for i, (token, score) in enumerate(zip(tokens, saliency_item['saliency'])):
                    if (i == len(tokens) - 1) or (token not in ['[CLS]', '[SEP]', '<s>', '</s>']):
                        if tmp_subword != ref_tokens[ref_id]:
                            tmp_subword_saliency_list.append(score)
                            tmp_subword += token if not token.startswith('##') else token[2:]
                        else:
                            translate_tokens.append(tmp_subword)
                            tmp_subword = token # reset
                            
                            translate_saliency.append(np.max(tmp_subword_saliency_list))
                            tmp_subword_saliency_list = [score] # reset

                            ref_id += 1                
                try:
                    assert translate_tokens == ref_tokens
                except:
                    print("[ERROR]")
                    print('translate tokens = ', translate_tokens)
                    print('ref = ', ref_tokens)
                    print('raw tokens = ', tokens)
                    exit()
                    

                # automatically identify rationales
                model_highlight_1 = []
                model_highlight_2 = []

                K = round(args.rationale_ratio * len(translate_tokens))
                top_k = np.argpartition(translate_saliency, -K)[-K:]
                top_k = [i for i in top_k if translate_saliency[i] >= args.min_saliency]
                for i in top_k:
                    if i >= len(premise.split()):
                        offset = len(premise.split())
                    else:
                        offset = 0
                    token = translate_tokens[i]
                    if offset == 0:
                        model_highlight_1.append(str(i))
                    else:
                        model_highlight_2.append(str(i - offset))
                item['model-highlighted-1'] = ','.join(model_highlight_1) if any(model_highlight_1) else '{}'
                item['model-highlighted-2'] = ','.join(model_highlight_2) if any(model_highlight_2) else '{}'                    
            else:
                item['model-highlighted-1'] = '{}'
                item['model-highlighted-2'] = '{}'
        
        with open(args.output_data_prefix + f'{split}_highlight.json', 'w') as f:
            json.dump(data, f, indent=2)


def extract_span_start_ends(ids):
    '''extract span from highlight ids'''
    if ids == '{}':
        return []
    else:
        ids = [int(i) for i in ids.split(',')]
        ids = sorted(ids)
        res = []
        start = ids[0]
        span_len = 1
        for i in range(1, len(ids)):
            if (ids[i] != ids[i-1] + 1): # new span
                res.append([start, start + span_len])
                start = ids[i]
                span_len = 1
            else: # consecutive span
                span_len += 1
        res.append([start, start + span_len]) # last span
        return res


def add_noise(text, mask_ids):
    tokens = text.split()
    if mask_ids is None:
        return tokens, []
    span_start_ends = extract_span_start_ends(mask_ids)
    start = 0
    no_noise_tokens = []
    noise_tokens = []
    for i, span in enumerate(span_start_ends):
        no_noise_tokens.append(tokens[start:span[0]] + ['extra_id_placeholder'])
        noise_tokens.append([f'extra_id_placeholder'] + tokens[span[0]:span[1]])
        start = span[1]        
    if start != len(tokens):
        no_noise_tokens.append(tokens[start:])
    no_noise_tokens = [x for y in no_noise_tokens for x in y]
    noise_tokens = [x for y in noise_tokens for x in y]
    return no_noise_tokens, noise_tokens   


def _prepare_item(item, start_idx, task='nli', no_label=False):
    inputs, label = [], []
    def update(text, mask_ids):
        nonlocal inputs
        nonlocal label
        tmp_input, tmp_label = add_noise(text, mask_ids)
        inputs += tmp_input
        label += tmp_label
    
    def replace_sentinel(start_sub_idx):
        # replace encode input
        sub_idx = start_sub_idx
        for i in range(len(inputs)):
            if inputs[i] == 'extra_id_placeholder':
                inputs[i] = f'<extra_id_{sub_idx}>'
                sub_idx += 1
        # replace decoder input
        sub_idx = start_sub_idx
        for i in range(len(label)):
            if label[i] == 'extra_id_placeholder':
                label[i] = f'<extra_id_{sub_idx}>'
                sub_idx += 1

    if task == 'nli':
        # text = f"mnli hypothesis: {hypothesis} premise: {premise} {item['label']}" 
        update('mnli hypothesis: ', None)
        update(item['hypothesis'], item['model-highlighted-2'])
        update('premise: ', None)
        update(item['premise'], item['model-highlighted-1'])
        if not no_label:
            update(item['label'], None)
        replace_sentinel(start_sub_idx=start_idx)
    else:
        raise Exception(f"not implemented task: {task}")

    return {"hypothesis": item['hypothesis'], "premise": item["premise"], "inputs": ' '.join(inputs), "label": ' '.join(label)}
    

def convert_highlight_to_mask(args):
    '''contruct data for text-to-text generation based on highlight''' 
    
    print('[INFO] convert highlight to mask')

    for split in ['train', 'dev']:
        with open(args.output_data_prefix + f'{split}_highlight.json', 'r') as f:
            data = json.load(f)
        '''
        reduce mask
        '''
        res = []
        cnt = 0
        for i in tqdm(data):
            ori_label = i['label']
            for label in LABELS:
                if args.cad and label == ori_label:
                    continue
                elif not args.cad and label != ori_label:
                    continue
                i['label'] = label
                if args.single:
                    for idx in range(1, 3):
                        item = deepcopy(i)
                        item[f'model-highlighted-{idx}'] = '{}'
                        tmp = _prepare_item(item, args.start_idx)
                        tmp['ori_label'] = ori_label
                        if tmp['inputs'] and tmp['label']:
                            res += [tmp] * args.num_sample
                else:
                    tmp = _prepare_item(i, args.start_idx)
                    tmp['ori_label'] = ori_label
                    if tmp['inputs'] and tmp['label']:
                        res += [tmp] * args.num_sample
        print(f'split = {split}, size = {len(res)}, cnt = {cnt}')

        suffix = ''
        if args.cad:
            suffix += '_cad'
        print('suffix = ', suffix)
        with open(args.output_data_prefix + f'{split}{suffix}.json', 'w') as f:
            json.dump(res, f, indent=2)


def convert_saliency_to_mask(args):
    convert_saliency_to_highlight(args)
    
    # get label-preserved rationale-masked data for training generator
    convert_highlight_to_mask(args)

    # get label-flipped rationale-masked data for generation
    args.cad = True
    convert_highlight_to_mask(args)

def parse():
    parser = argparse.ArgumentParser(description='finetune bert')

    # data  
    parser.add_argument("--input_data_prefix", type=str, default=f'../data/snli/ori_data/')
    parser.add_argument("--output_data_prefix", type=str, default=f'../data/snli/rationale_mask/')
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--start_idx", type=int, default=0, help='start idx of mask token')
   
    # cad
    parser.add_argument("--single", action='store_true')
    parser.add_argument("--cad", action='store_true', default=False)
    parser.add_argument("--min_confidence", default=0.8, type=float)
    parser.add_argument("--min_saliency", default=0.3, type=float)
    parser.add_argument("--rationale_ratio", default=0.3, type=float)
   
    # nlu model
    parser.add_argument("--nlu_model", type=str, default='roberta-large')

    args = parser.parse_args()

    print(args)

    os.makedirs(args.output_data_prefix, exist_ok=True)

    return args

if __name__ == '__main__':

    args = parse()

    convert_saliency_to_mask(args)
