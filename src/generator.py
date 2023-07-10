import argparse
import os
import json
import jsonlines
import math
from tqdm import tqdm
import numpy as np
import random
import re
from pathlib import Path
from copy import copy
from typing import Optional, Union


import nltk
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import T5TokenizerFast, T5ForConditionalGeneration, AutoConfig, logging, get_linear_schedule_with_warmup
from transformers import LogitsProcessorList, StoppingCriteriaList
from transformers.generation_utils import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


TASK_CONFIG={
    "nli": {
        "label_map": {"contradiction": 0, "neutral": 1, "entailment": 2}, 
        "pair": True},
    "sst": {
        "label_map": {"negative": 0, "positive": 1}, 
        "pair": False},
    }


logging.enable_explicit_format()
class Helper():
    def __init__(self, args):
        self.tokenizer = T5TokenizerFast.from_pretrained(args.model_config, local_files_only=True)
        # print(self.tokenizer.extra_ids)
        self.extra_id_start = self.tokenizer.convert_tokens_to_ids(['<extra_id_0>'])[0]
        assert self.extra_id_start ==  32099
        self.add_tokens()  # set all special tokens

    def add_tokens(self):
        pass

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_vocab_size(self):
        return len(self.tokenizer)

class CadDataset(Dataset):
    def __init__(self, split, path, task, args):
        super().__init__()
        self.split = split
        self.path = path
        self.task = task
        self.args = args

        self.label_map = TASK_CONFIG[self.task]['label_map']
        self.pair = TASK_CONFIG[self.task]['pair']
        self.num_labels = len(set(self.label_map.values()))
        self.data = self.load_data(path)

    def add_noise(self, text, mask_ratio, mask_mean_len):
        tokens = text.split()
        if mask_ratio == 0:
            return tokens, []
        elif mask_ratio == 1:
            return ['extra_id_placeholder'], ['extra_id_placeholder'] + tokens
        else:
            span_start_ends = self._random_spans_noise_mask(len(tokens), mask_ratio, mask_mean_len)
            start = 0
            no_noise_tokens = []
            noise_tokens = []
            for i, span in enumerate(span_start_ends):
                no_noise_tokens.append(tokens[start:span[0]] + [f'extra_id_placeholder']) # 0: placeholder for sentinel idx
                noise_tokens.append([f'extra_id_placeholder'] + tokens[span[0]:span[1]])
                start = span[1]
            if start != len(tokens):
                no_noise_tokens.append(tokens[start:])
            no_noise_tokens = [x for y in no_noise_tokens for x in y]
            noise_tokens = [x for y in noise_tokens for x in y]
            return no_noise_tokens, noise_tokens

    def _random_spans_noise_mask(self, length, noise_density, mean_noise_span_length=2.0):
        num_noise_tokens = round(length * noise_density)
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        def random_segment(seq_length, num_segment):
            x = (torch.arange(seq_length - 1) < (num_segment - 1)).long()
            a = torch.randperm(seq_length - 1)
            x = x[a]
            x = F.pad(x, [1, 0])
            segment_id = torch.cumsum(x, dim=0)
            segment_lengths = torch.zeros(num_segment, dtype=torch.long).scatter_add_(0, segment_id, torch.ones(seq_length, dtype=torch.long))
            return segment_lengths

        noise_span_lengths = random_segment(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = random_segment(num_nonnoise_tokens, num_noise_spans)
        if random.random() < 0.5:
            interleaved_span_lengths = torch.stack([nonnoise_span_lengths, noise_span_lengths], dim=1).view(num_noise_spans * 2)
        else:
            interleaved_span_lengths = torch.stack([noise_span_lengths, nonnoise_span_lengths], dim=1).view(num_noise_spans * 2)
            interleaved_span_lengths = torch.cat([torch.tensor([0]), interleaved_span_lengths[:-1]], dim=0)

        span_start_ends = torch.cumsum(interleaved_span_lengths, dim=0).view(-1, 2)
        return span_start_ends.tolist()


    def prepare(self, item):      
        inputs, label = [], []

        def update(text, mask_ratio, mask_mean_len=None):
            nonlocal inputs
            nonlocal label
            tmp_input, tmp_label = self.add_noise(text, mask_ratio, mask_mean_len)
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

        if self.task == 'nli':
            # text = f"mnli hypothesis: {hypothesis} premise: {premise} {item['label']}" 
            update('mnli hypothesis: ', 0)
            update(item['hypothesis'], self.args.mask_ratio, self.args.mask_mean_len)
            update('premise: ', 0)
            update(item['premise'], self.args.mask_ratio, self.args.mask_mean_len)
            update(item['label'], 0)
            replace_sentinel(start_sub_idx=0)
        else:
            raise Exception(f"not implemented task: {self.task}")

        return {"ori_label": item['label'], "inputs": ' '.join(inputs), "label": ' '.join(label)}


    def load_data(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        if self.args.random_mask:
            data = [self.prepare(i) for i in data]
        return data
    
    def __len__(self):
        return len(self.data)

    def show_example(self):
        print(self.data[0])
        tmp = self.__getitem__(0)
        print(tmp)

    def create_perturb_input(self, raw_input, ori_label):
        assert raw_input.endswith(ori_label)
        perturb_input = []
        for new_label in self.label_map:
            if new_label == ori_label:
                continue
            tmp = raw_input[: -len(ori_label)] + new_label
            perturb_input.append(tmp)
        return perturb_input

    def __getitem__(self, idx):
        raw_item = self.data[idx]

        ori_label, raw_input, raw_label = raw_item['ori_label'], raw_item['inputs'], raw_item['label']
        # item = helper.tokenizer(raw_input)
        # item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long)
        # item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.float)
        # item['label'] = torch.tensor(helper.tokenizer(raw_label)['input_ids'])
        if self.split != 'test':
            item = {}
            item['ori_input'] = raw_input
            item['perturb_input'] = self.create_perturb_input(raw_input, ori_label)
            item['label'] = raw_label
        else:
            item = helper.tokenizer(raw_input)
            item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long)
            item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.float)
            item['label'] = torch.tensor(helper.tokenizer(raw_label)['input_ids'])
        return item


def pad_collate(batch, max_len=128):
    ori_input_list = []
    ori_label_list = []
    perturb_input_list = []
    perturb_label_list = []
    for item in batch:
        label = helper.tokenizer(item['label'])['input_ids'][:max_len]
        ori_input_list.append(item['ori_input'])
        ori_label_list.append(torch.tensor(label, dtype=torch.long))
        for j in item['perturb_input']:
            perturb_input_list.append(j)
            perturb_label_list.append(torch.tensor(label, dtype=torch.long))
            
    input_list = ori_input_list + perturb_input_list
    label_list = ori_label_list + perturb_label_list

    res = {}
    encoded_inputs = helper.tokenizer(input_list, return_tensors='pt', padding="max_length", max_length=max_len, truncation=True)
    res['input_ids'] = encoded_inputs.input_ids
    res['attention_mask'] = encoded_inputs.attention_mask
    res['labels'] = pad_sequence(label_list, batch_first=True, padding_value=-100)
    
    ori_batch_size = len(batch)
    new_batch_size = len(input_list)
    return res


def pad_collate_generation(batch, max_len=128):
    res = {}
    res['input_ids'] = pad_sequence([x['input_ids'][:max_len] for x in batch], batch_first=True,
                                    padding_value=helper.tokenizer.pad_token_id)
    res['attention_mask'] = pad_sequence([x['attention_mask'][:max_len] for x in batch], batch_first=True,
                                         padding_value=0)
    if 'label' in batch[0]:
        res['labels'] = pad_sequence([x['label'][:max_len] for x in batch], batch_first=True,
                                    padding_value=-100)
    return res


class T5(pl.LightningModule):
    def __init__(self, load_dir=None, lr=None, weight_decay=None, warm_up=None, model_config=None, args=None):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warm_up = warm_up
        self.max_step = None
        
        self.args = args

        self.label_map = TASK_CONFIG[self.args.task]['label_map']
        self.num_per_sample = len(self.label_map) # 每个样本的实际数量: 正样本 + 负样本
        print('num per sample = ', self.num_per_sample)

        self.save_hyperparameters()
        print("load dir = ", load_dir)
        print('model config = ', model_config)
        if load_dir is None: # load pretrain parameter and config
            self.model = T5ForConditionalGeneration.from_pretrained(model_config)
        else: # only load config
            config = AutoConfig.from_pretrained(model_config)
            self.model = T5ForConditionalGeneration.from_pretrained(config)

        if self.args.gradient_checkpointing:
            if self.args.train:
                self.model.config.use_cache = False # gradient checkpointing is incompatible with use_cache during decoder training
            self.model.gradient_checkpointing_enable()
        
    def forward(self, **kargs):
        return self.model(**kargs)

    def unlikelyhood_ce_loss(self, logit, label, reduction='mean'):
        # m, _ = torch.max(logit, dim=1, keepdim=True)
        # exp_logit = torch.exp(logit - m)
        exp_logit = torch.softmax(logit, dim=1)
        sum_exp_logit = torch.sum(exp_logit, dim=1)

        eps = 1e-20
      
        num = (sum_exp_logit - exp_logit[torch.arange(exp_logit.shape[0]), label]) # (sum - p)
        num = torch.log(num + eps) # (sum - p + eps)
        denon = torch.log(sum_exp_logit + eps) # (sum + eps)
       
        # Negative log probability

        loss = -(num - denon)
        loss_mask = (1 < label) & (label < helper.extra_id_start)
        loss = loss[loss_mask]
        
        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()    
        
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels) # 用labels构造decoder_input_ids, 不使用他计算的loss
        logits = output.logits
        # logits, loss = output.logits, output.loss
        batch_size = input_ids.size(0) // self.num_per_sample
        ori_input_logits, perturb_input_logits = logits[: batch_size, :, :].view(-1, logits.size(-1)), logits[batch_size: , :, :].view(-1, logits.size(-1))
        ori_labels, perturb_labels = labels[: batch_size, :].view(-1), labels[batch_size: , :].view(-1)

        likelyhood_loss = F.cross_entropy(ori_input_logits, ori_labels, reduction='mean')
        unlikelyhood_loss = self.unlikelyhood_ce_loss(perturb_input_logits, perturb_labels, reduction='mean')

        if self.args.multi_mode == 'sum':
            loss = likelyhood_loss +  self.args.alpha * unlikelyhood_loss
        else:
            raise Exception(f"unimplemented multi-task mode : {self.args.multi_mode}")

        self.log('train_likelihood_loss', likelyhood_loss.item(), on_step=True, prog_bar=True)
        self.log('train_unlikelihood_loss', unlikelyhood_loss.item(), on_step=True, prog_bar=True)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = output.logits

        batch_size = input_ids.size(0) // self.num_per_sample
        ori_input_logits, perturb_input_logits = logits[: batch_size, :, :].view(-1, logits.size(-1)), logits[batch_size: , :, :].view(-1, logits.size(-1))
        ori_labels, perturb_labels = labels[: batch_size, :].view(-1), labels[batch_size: , :].view(-1)

        likelyhood_loss = F.cross_entropy(ori_input_logits, ori_labels, reduction='mean')
        unlikelyhood_loss = self.unlikelyhood_ce_loss(perturb_input_logits, perturb_labels, reduction='mean')


        if self.args.multi_mode == 'sum':
            loss = likelyhood_loss + self.args.alpha * unlikelyhood_loss
        elif self.args.multi_mode == 'norm':
            pass
        else:
            raise Exception(f"unimplemented multi-task mode : {self.args.multi_mode}")

        self.log('val_likelihood_loss', likelyhood_loss.item(), on_epoch=True, sync_dist=True)
        self.log('val_unlikelihood_loss',unlikelyhood_loss.item(), on_epoch=True, sync_dist=True)
        self.log('val_loss', loss.item(), on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits, loss = output.logits, output.loss
        self.log('test_loss', loss.item(), on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        betas = (0.9, 0.98)
        if self.args.optimizer == 'adamw':
            if self.args.deepspeed:
                optimizer = FusedAdam(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay, betas=betas)
            else:
                optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay, betas=betas)
            if self.warm_up:
                num_warmup_steps = int(self.max_step * self.warm_up)
                num_training_steps = self.max_step
                print("num_warmup_steps = ", num_warmup_steps)
                print("num_training_steps = ", num_training_steps)
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps) # num_warmup_steps, num_training_steps
                # # scheduler = get_constant_schedule_with_warmup(optimizer, self.max_step * self.warmup)
                return (
                    [optimizer],
                    [
                        {
                            'scheduler': scheduler,
                            'interval': 'step',
                            'frequency': 1,
                            'reduce_on_plateau': False,
                        }
                    ]
                )
            else:
                return optimizer
        elif self.args.optimizer == 'adafactor':
            from transformers.optimization import Adafactor, AdafactorSchedule
            optimizer = Adafactor(optimizer_grouped_parameters, lr=self.lr, scale_parameter=False, relative_step=False)
            return optimizer


def train(args):
    train_set = CadDataset('train', args.train_set, args.task, args)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=pad_collate)

    dev_set = CadDataset('dev', args.dev_set, args.task, args)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=pad_collate)
    
    print('train_size = ', len(train_set))
    print('dev_size = ', len(dev_set))
    train_set.show_example()
    dev_set.show_example()

    model = T5(model_config=args.model_config, load_dir=args.load_dir, lr=args.lr, warm_up=args.warm_up, weight_decay=args.weight_decay, args=args)
    model.max_step = math.ceil(len(train_set) / args.batch_size) * args.max_epochs

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', verbose=True, save_on_train_epoch_end=False)

    # earlystop_callback = EarlyStopping(monitor='val_loss', verbose=True, mode='min')
    plugins = DeepSpeedPlugin(stage=2, logging_level=logging.INFO) if args.deepspeed else None

    args.eval_step = min(args.eval_step, math.ceil(len(train_set) / (args.batch_size * int(args.gpus))))
    trainer = pl.Trainer(gpus=int(args.gpus), max_epochs=args.max_epochs,
                         callbacks=[checkpoint_callback], val_check_interval=args.eval_step, accumulate_grad_batches=args.accumulation_step,
                         default_root_dir=args.save_dir, strategy=plugins, precision=args.precision)
    trainer.fit(model, train_dataloader, dev_dataloader)


def test(args):
    test_set = CadDataset('test', args.test_set, args.task)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate)
    model = T5.load_from_checkpoint(args.load_dir, model_config=args.model_config, load_dir=args.load_dir)
    model.eval()
    trainer = pl.Trainer(gpus=[int(args.gpus)], checkpoint_callback=False, logger=False) # disable logging
    result = trainer.test(model, test_dataloader, verbose=True)
    log_path = args.base_load_dir + "/test_result.json"
    test_result = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            test_result = json.load(f)
    test_set_name = args.test_set.split('/')[-2]
    test_result[test_set_name] = result    
    with open(log_path, 'w') as f:
        json.dump(test_result, f, indent=2)


def generate(args):

    def postprocess(token_list):
        token_list = [i for i in token_list if i > 0]
        string = helper.tokenizer.decode(token_list)
        string = string.replace("</s>", "").replace("<s>", "").replace("<pad>", "").strip()
        return string

    def fill_in_the_blank(context, generate):
        def my_split(text):
            tokens = re.split('(<extra_id_.?>)', text)
            return [i.strip(' ') for i in tokens if i]
        ori_context = copy(context)
        ori_generate = copy(generate)
        context = my_split(context) 
        generate = my_split(generate)
        if len(context) -1 > len(generate):
            return None
        j = 0
        flag = True
        for i in range(len(context)):
            if context[i].startswith('<extra_id_'):
                if generate[j] != context[i]:
                    flag = False
                    break
                else:
                    context[i] = generate[j+1]
                    j += 2
        if flag == False:
            print('ori_context = ', ori_context)
            print('ori_generate = ', ori_generate)
            print('context = ', context)
            print('generate = ', generate)
            print('*'*50)

        if not flag:
            return None
        else:
            return ' '.join(context)

    def extract(text):
        tmp = text.split(' ')
        text, label = ' '.join(tmp[:-1]), tmp[-1]
        
        if args.task == 'nli':
            start_premise = text.index('premise')
            hypothesis = text[len("mnli hypothesis: "): start_premise].strip()
            premise = text[start_premise + len('premise:'):].strip()
            return premise, hypothesis, label
        elif args.task == 'sst':
            text = text[len("sst2 sentence: "):].strip()
            return text, label
        else:
            raise NotImplementedError(f"not implemented task: {args.task}")

    test_set = CadDataset("test", args.test_set, args.task, args)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, collate_fn=pad_collate_generation)
    device = torch.device(f"cuda:{args.gpus}")
    if args.load_dir is not None:
        model = T5.load_from_checkpoint(args.load_dir, model_config=args.model_config, args=args, strict=True)
    else:
        model = T5(model_config=args.model_config, args=args)

    model.to(device)
    model.eval()

    result = []

    error_logs = []

    offset = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            dec_input_ids = torch.full((input_ids.shape[0], 1), model.model.config.decoder_start_token_id).to(device)
            bad_word_ids = helper.tokenizer(['neutral', 'entailment', 'contradiction'], add_special_tokens=False).input_ids
            gen = model.model.generate(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=dec_input_ids, do_sample=True, top_p=0.9, temperature=0.7, max_length=args.max_gen_len, bad_words_ids=bad_word_ids)
            for idx, (context_ids, generate_ids, span_label_ids) in enumerate(zip(batch['input_ids'], gen, batch['labels'])):
                context = test_set.data[offset + idx]['inputs']
                generate = postprocess(generate_ids)
                span_label = postprocess(span_label_ids)
                filled_text = fill_in_the_blank(context, generate)
                if filled_text is None:
                    result.append(None) # placeholder, to align with input data
                    error_logs.append({"context": context, "generate": generate, "span_label": span_label, "raw_context": helper.tokenizer.decode(context_ids), "raw_generate": helper.tokenizer.decode(generate_ids)})
                    continue
                if args.task == 'nli':
                    try:
                        premise, hypothesis, label = extract(filled_text)
                    except:
                        print('context = ', context)
                        print('generate = ', generate)
                        print('filled_text = ', filled_text)
                    result.append({'input': context, 'generate': generate, 'premise': premise, 'hypothesis': hypothesis, "label": label, "span_label": span_label})
                elif args.task == 'sst':
                    text, label = extract(filled_text)
                    result.append({"text": text, "label": label, "span_label": span_label})
          
            offset += input_ids.shape[0]

    final_result = []
    for input_data, gen_data in zip(test_set.data, result):
        if gen_data is None: # filter None
            continue
        if args.task == 'nli':
            if 'ori_label' in input_data:
                gen_data['ori_label'] = input_data['ori_label']
            if 'premise' in input_data:
                gen_data['ori_premise'] = input_data['premise']
                gen_data['ori_hypothesis'] = input_data['hypothesis']
                gen_data['ori_label'] = input_data['ori_label']
        elif args.task == 'sst':
            if 'text' in input_data:
                gen_data['ori_text'] = input_data['text']
                gen_data['ori_label'] = input_data['ori_label']
        else:
            raise NotImplementedError(f"not implemented task: {args.task}")
        final_result.append(gen_data)
    
    if args.load_dir is not None:
        base_load_dir = Path(args.load_dir).parent
        save_path = base_load_dir / args.save_file
    else:
        save_path = args.save_file
    print('save_path = ', save_path)
    with open(save_path, 'w') as f:
        json.dump(final_result, f, indent=2)

    if args.load_dir is not None:
        error_save_file = args.save_file.split('_')[0] + '_log.json'
        error_save_path = base_load_dir / error_save_file
        from numpyencoder import NumpyEncoder
        with open(error_save_path, 'w') as f:
            json.dump(error_logs, f, indent=2, cls=NumpyEncoder)


def parse():
    parser = argparse.ArgumentParser(description='finetune bert')

    # data
    parser.add_argument("--task", type=str, default="nli",
                        help="task type, [nli, sst]")
    parser.add_argument('--train_set', type=str, 
                        help='Path of training set')
    parser.add_argument('--dev_set', type=str,
                        help='Path of validation set')
    parser.add_argument('--test_set', type=str,
                        help='Path of test set')
    parser.add_argument("--save_file", type=str, default='generate.json')

    # device
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of cpu cores to use')
    parser.add_argument('--gpus', default=1, help='num gpus')
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--deepspeed", action='store_true')
    parser.add_argument("--accumulation_step", type=int, default=None)
    
    # unlikelihood
    parser.add_argument("--multi_mode", type=str, default='sum', help='sum, norm')
    parser.add_argument("--alpha", type=float, default=0)
    parser.add_argument("--random_mask", action='store_true')
    parser.add_argument("--mask_ratio", type=float, default=0.3)
    parser.add_argument("--mask_mean_len", type=int, default=2)

    # hyperparameters
    parser.add_argument("--seed", type=int, default=20210826, 
                        help='random seed')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help='weight decay')
    parser.add_argument('--warm_up', type=float, default=0,
                        help='warm-up rate')
    parser.add_argument('--max_epochs', type=int, default=5,
                        help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of mini-batch')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle data')
    parser.add_argument('--eval_step', type=int, default=1000)
    
    # model
    parser.add_argument("--model_config", type=str, default='bert-base-uncased',
                        help="config path for model(tokenizer)")
    parser.add_argument("--optimizer", type=str, default='adamw', help='adam, adafactor')

    # generate
    parser.add_argument("--max_gen_len", type=int, default=20, help='max generate len')
    parser.add_argument("--gen_times", type=int, default=1, help='generation times for each sample')
    
    # model load/save
    parser.add_argument('--load_dir', type=str, default=None,
                        help='Directory of checkpoint to load for predicting')
    parser.add_argument('--save_dir', type=str,
                        help='Path to save model')
    parser.add_argument('--output_path', type=str,
                        help='saliency analysis result')

    # mode
    parser.add_argument('--train', action='store_true', help='train')
    parser.add_argument('--generate', action='store_true',
                        help='predict result')
    parser.add_argument('--test', action='store_true',
                        help='test')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse()
    for k, v in vars(args).items():
        print(f"{k}:\t{v}")
        
    helper = Helper(args)

    pl.seed_everything(args.seed)

    if args.generate:
        generate(args)
    elif args.test:
        test(args)
    else:
        train(args)
