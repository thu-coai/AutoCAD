import argparse
import os
import json
import math
from tqdm import tqdm
import numpy as np
import random
from pathlib import Path
from copy import copy

from numpyencoder import NumpyEncoder

import nltk
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, logging, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import torch
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
        print('args.model_config = ', args.model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_config, local_files_only=True)
        self.mask_id = self.get_token_id(self.tokenizer.mask_token)
        self.add_tokens()  # set all special tokens

    def add_tokens(self):
        self.tokenizer.add_special_tokens({"pad_token": '[PAD]'})

    def get_token_id(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def get_vocab_size(self):
        return len(self.tokenizer)

class CadDataset(Dataset):
    def __init__(self, split, path, task, noise_ratio=None, noise_source=None, noise_target=None, erase=False):
        super().__init__()
        self.split = split
        self.path = path
        self.task = task

        self.noise_ratio = noise_ratio
        self.noise_source = noise_source
        self.noise_target = noise_target
        
        self.erase=erase

        self.label_map = TASK_CONFIG[self.task]['label_map']
        self.pair = TASK_CONFIG[self.task]['pair']
        self.num_labels = len([i for i in self.label_map.values() if i >= 0])
        self.data = self.load_data(path)

    def load_data(self, path):
        
        with open(path, 'r') as f:
            raw_data = json.load(f)
        data = [i for i in raw_data if i['label'] in self.label_map]
        # assert len(raw_data) == len(data)
        for i in data:
            i['label'] = self.label_map[i['label']]

        if self.erase == True:
            erased_data = []
            for idx, ori_item in enumerate(data):
                if self.pair:
                    offset = 0
                    base_item = copy(ori_item)
                    base_item['mask_id'] = None
                    base_item['id'] = idx
                    erased_data.append(base_item)

                    for key in ['premise', 'hypothesis']:
                        res = self.mask_erase(ori_item[key])
                        for erase_item in res:
                            new_item = copy(ori_item)
                            new_item[key] = erase_item['text']
                            new_item['mask_id'] = erase_item['mask_id'] + offset
                            new_item['id'] = idx
                            erased_data.append(new_item)
                        offset += res[-1]['mask_id'] + 1
                else:
                    base_item = copy(ori_item)
                    base_item['mask_id'] = None
                    base_item['id'] = idx
                    erased_data.append(base_item)
                    res = self.mask_erase(ori_item['text'])
                    for erase_item in res:
                        new_item = copy(ori_item)
                        new_item['text'] = erase_item['text']
                        new_item['mask_id'] = erase_item['mask_id']
                        new_item['id'] = idx
                        erased_data.append(new_item)

            return erased_data

        else:
            if self.noise_ratio:
                for idx, i in enumerate(data):
                    if self.pair:
                        data[idx]['premise'] = self.mask(i['premise'], i[f'{self.noise_source}-1'])
                        data[idx]['hypothesis'] = self.mask(i['hypothesis'], i[f'{self.noise_source}-2'])
                    else:
                        data[idx]['text'] = self.mask(i['text'], i[self.noise_source])
                    
            return data
    
    def __len__(self):
        return len(self.data)

    def show_example(self):
        print(self.data[0])
        print(self.__getitem__(0))

    def mask_erase(self, text):
        tokens = text.split()
        output = []
        for id in range(len(tokens)):
            item = copy(tokens)
            item[id] = helper.tokenizer.mask_token
            output.append({
                "text": ' '.join(item),
                "mask_id": id
            })
        return output

    def mask(self, text, loc):
        text = text.split()
        empty = False

        try:
            raw_loc = [int(i) for i in loc.split(",")]
        except:
            empty = True

        if not empty:
            if self.noise_target == 'rationales':
                loc = raw_loc
            else:
                loc = list(set(range(len(text)))- set(raw_loc))
            for i in range(len(text)):
                if i in loc:
                    if random.random() < self.noise_ratio:
                        text[i] = helper.tokenizer.mask_token
            return ' '.join(text)
        else:
            if self.noise_target == 'rationales':
                return ' '.join(text)
            else:
                return ' '.join([helper.tokenizer.mask_token] * len(text))            


    def __getitem__(self, idx):
        ori_item = self.data[idx]
        if self.pair:
            item = helper.tokenizer(ori_item['premise'], ori_item['hypothesis'])
        else:
            item = helper.tokenizer(ori_item['text'])

        item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long)
        item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.float)
        if 'token_type_ids' in item:
            item['token_type_ids'] = torch.tensor(item['token_type_ids'], dtype=torch.long)
        if 'label' in ori_item:
            item['label'] = torch.tensor(ori_item['label'], dtype=torch.long)
        return item


def pad_collate(batch, max_len=128):
    res = {}
    res['input_ids'] = pad_sequence([x['input_ids'][:max_len] for x in batch], batch_first=True,
                                    padding_value=helper.tokenizer.pad_token_id)
    res['attention_mask'] = pad_sequence([x['attention_mask'][:max_len] for x in batch], batch_first=True,
                                         padding_value=0)
    if 'token_type_ids' in batch[0]:
        res['token_type_ids'] = pad_sequence([x['token_type_ids'][:max_len] for x in batch], batch_first=True,
                                        padding_value=0)
    if 'label' in batch[0]:
        res['labels'] = torch.stack([x['label'] for x in batch], dim=0)
    return res

class Bert(pl.LightningModule):
    def __init__(self, load_dir=None, lr=None, weight_decay=None, warm_up=None, num_labels=None, model_config=None):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay
        self.warm_up = warm_up
        self.max_step = None

        self.save_hyperparameters()
        
        if load_dir is None: # load pretrain parameter and config
            self.model = AutoModelForSequenceClassification.from_pretrained(model_config, num_labels=num_labels)
        else: # only load config
            config = AutoConfig.from_pretrained(model_config, num_labels=num_labels)
            self.model = AutoModelForSequenceClassification.from_config(config)

        self.model.resize_token_embeddings(helper.get_vocab_size())

        print('num_labels = ', self.model.num_labels)

    def forward(self, **kargs):
        return self.model(**kargs)

    def counterfactual_ce_loss(self, logit, y, reduction='none'):
        assert (y < 0).all(), str(y)

        cf_y = -(y + 1)

        if logit.shape[1] == 2: # 2-cls
            return F.cross_entropy(logit, 1 - cf_y, reduction=reduction)

        # Implement my own logsumexp trick
        m, _ = torch.max(logit, dim=1, keepdim=True)
        exp_logit = torch.exp(logit - m)
        sum_exp_logit = torch.sum(exp_logit, dim=1)

        eps = 1e-20
        num = (sum_exp_logit - exp_logit[torch.arange(exp_logit.shape[0]), cf_y])
        num = torch.log(num + eps)
        denon = torch.log(sum_exp_logit + eps)

        # Negative log probability
        loss = -(num - denon)

        if reduction == 'none':
            return loss
        if reduction == 'mean':
            return loss.mean()
        if reduction == 'sum':
            return loss.sum()

    def counterfactual_criterion(self, logits, y):
        alpha = 2
        if torch.all(y >= 0):
            return F.cross_entropy(logits, y, reduction='mean')
        if torch.all(y < 0):
            return alpha * self.counterfactual_ce_loss(logits, y, reduction='mean')

        loss1 = F.cross_entropy(logits[y >= 0], y[y >= 0], reduction='sum')
        loss2 = self.counterfactual_ce_loss(logits[y < 0], y[y < 0], reduction='sum')
        return loss1 / logits.shape[0] + alpha * loss2 / logits.shape[0]

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch else None
        labels = batch['labels']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        logits = output.logits
        loss = output.loss
        # loss = self.counterfactual_criterion(logits, labels)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch else None
        labels = batch['labels']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        logits = output.logits
        # loss = self.counterfactual_criterion(logits, labels)
        loss = output.loss
        self.log('val_loss', loss.item())

        # logits, loss = output.logits, output.loss
        preds = logits.argmax(-1)
        acc = (preds == labels[labels>=0]).float().mean().item()

        # self.log('val_loss', loss.item())
        self.log('val_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch else None
        labels = batch['labels']
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        logits, loss = output.logits, output.loss
        preds = logits.argmax(-1)
        acc = (preds == labels).float().mean().item()

        self.log('test_loss', loss.item())
        self.log('test_acc', acc, on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        # print('params = ', [name for name, param in self.named_parameters()])
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
        if self.warm_up:
            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay, betas=betas)
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
            return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay, betas=betas)


def train(args):
    train_set = CadDataset('train', args.train_set, args.task, noise_ratio=args.noise_ratio, noise_source=args.noise_source, noise_target=args.noise_target)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=pad_collate)

    dev_set = CadDataset('dev', args.dev_set, args.task, noise_ratio=args.noise_ratio, noise_source=args.noise_source, noise_target=args.noise_target)
    dev_dataloader = DataLoader(dev_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers,
                                collate_fn=pad_collate)
    
    print('train_size = ', len(train_set))
    print('dev_size = ', len(dev_set))
    train_set.show_example()

    model = Bert(model_config=args.model_config, load_dir=args.load_dir, lr=args.lr, warm_up=args.warm_up, weight_decay=args.weight_decay, num_labels=train_set.num_labels)
    model.max_step = math.ceil(len(train_set) / args.batch_size) * args.max_epochs

    checkpoint_callback = ModelCheckpoint(monitor='val_acc', save_top_k=1, mode='max', verbose=True, save_on_train_epoch_end=False)
    val_check_interval = 1.0
    trainer = pl.Trainer(gpus=[int(args.gpus)], max_epochs=args.max_epochs,
                         callbacks=[checkpoint_callback], val_check_interval=val_check_interval,
                         default_root_dir=args.save_dir, )
    trainer.fit(model, train_dataloader, dev_dataloader)


def test(args):
    test_set = CadDataset('test', args.test_set, args.task)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate)
    model = Bert.load_from_checkpoint(args.load_dir, model_config=args.model_config, load_dir=args.load_dir, num_labels=test_set.num_labels)
    model.eval()
    trainer = pl.Trainer(gpus=[int(args.gpus)], checkpoint_callback=False, logger=False) # disable logging
    result = trainer.test(model, test_dataloader, verbose=True)
    
    base_load_dir = Path(args.load_dir).parent
    log_path = base_load_dir / "test_result.json"
    test_result = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            test_result = json.load(f)
    test_set_name = args.test_set.split('/')[-2]
    test_result[test_set_name] = result    
    with open(log_path, 'w') as f:
        json.dump(test_result, f, indent=2)


def get_saliency_erase(args):
    test_set = CadDataset("test", args.test_set, args.task, erase=True)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate)
    device = torch.device(f"cuda:{args.gpus}")
    model = Bert.load_from_checkpoint(args.load_dir, model_config=args.model_config, load_dir=args.load_dir, num_labels=test_set.num_labels)
    model.to(device)
    model.eval()

    embedding = model.model.get_input_embeddings()

    result = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch else None
            labels = batch['labels'].to(device)

            mask_pos = input_ids == helper.mask_id
            # print('mask_pos = ', mask_pos)
            input_embeds = embedding(input_ids)
            # print('shape of embeds = ', input_embeds.shape)
            input_embeds[mask_pos] = 0
            # print('erased shape of embeddings = ', input_embeds.shape)
            outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            # logits = torch.softmax(logits, dim=-1).tolist()
            probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            # scores = logits[:, 1].tolist()
            for sample, score, label, pred in zip(batch['input_ids'], probs, batch['labels'].numpy(), preds):
                result.append({"score": score, "pred": pred, "label": label})

    final_result = []
    for input_data, predict_data in zip(test_set.data, result):
        if args.task == 'nli':
            predict_data['premise'] = input_data['premise']
            predict_data['hypothesis'] = input_data['hypothesis']
        elif args.task == 'sst':
            predict_data['text'] = input_data['text']
        predict_data['id'] = input_data['id']
        predict_data['mask_id'] = input_data['mask_id']

        final_result.append(predict_data)

    ids = set([i['id'] for i in final_result])
    print('size of ids = ', len(ids))

    merge_dict = {i:[] for i in ids}
    for item in final_result:
        merge_dict[item['id']].append(item)

    output = []
    for i in merge_dict.values():
        saliency_list = []
        base_item = i[0]
        label = base_item['label']
        base_score = base_item['score'][label]
        for mask_id, j in enumerate(i[1:]):
            assert j['mask_id'] == mask_id
            erase_score = j['score'][label]
            saliency = base_score - erase_score
            saliency_list.append(saliency)
        
        base_item['saliency'] = saliency_list
        output.append(base_item)

    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)



def predict(args):
    test_set = CadDataset("test", args.test_set, args.task)
    print('test_size = ', len(test_set))
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate)
    device = torch.device(f"cuda:{args.gpus}")
    model = Bert.load_from_checkpoint(args.load_dir, model_config=args.model_config, load_dir=args.load_dir, num_labels=test_set.num_labels)
    model.to(device)
    model.eval()

    result = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch else None
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits
            # logits = torch.softmax(logits, dim=-1).tolist()
            probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            # scores = logits[:, 1].tolist()
            for sample, score, label, pred in zip(batch['input_ids'], probs, batch['labels'].numpy(), preds):
                result.append({"score": score, "pred": pred, "label": label})
                    # f.write(str(score) + '\n')

    # labels = np.array([i['label'] for i in result])
    # preds = np.array([i['pred'] for i in result])
    # print("acc = ", np.mean((labels==preds)))
    
    final_result = []
    for input_data, predict_data in zip(test_set.data, result):
        if args.task == 'nli':
            predict_data['premise'] = input_data['premise']
            predict_data['hypothesis'] = input_data['hypothesis']
            if 'ori_label' in input_data:
                predict_data['ori_label'] = input_data['ori_label']
            if 'ori_premise' in input_data:
                predict_data['ori_premise'] = input_data['ori_premise']
                predict_data['ori_hypothesis'] = input_data['ori_hypothesis']
                predict_data['span_label'] = input_data['span_label']
        elif args.task == 'sst':
            predict_data['text'] = input_data['text']
            if 'ori_label' in input_data:
                predict_data['ori_label'] = input_data['ori_label']
                predict_data['ori_text'] = input_data['ori_text']
        final_result.append(predict_data)

    save_path = args.predict_file
    with open(save_path, 'w') as f:
        json.dump(final_result, f, indent=2, cls=NumpyEncoder)


def get_saliency(args):
    import heapq
    
    test_set = CadDataset("test", args.test_set, args.task)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, collate_fn=pad_collate)
    device = torch.device(f"cuda:{args.gpus}")
    model = Bert.load_from_checkpoint(args.load_dir, model_config=args.model_config, load_dir=args.load_dir, num_labels=test_set.num_labels)
    model.to(device)
    model.eval() # turn off dropout

    optimizer = torch.optim.AdamW(model.parameters())
    embedding = model.model.get_input_embeddings()
    result = []

    K = 10
    for i, batch in tqdm(enumerate(test_dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
        labels = batch['labels'].to(device)
        input_embeds = embedding(input_ids)
        input_embeds.retain_grad()
        output = model(inputs_embeds=input_embeds, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
        loss, logits = output.loss, output.logits
        optimizer.zero_grad()
        loss.backward()

        bc = input_ids.shape[0]
        if args.saliency_mode == 'gradient':
            unary = torch.sqrt(torch.mean(input_embeds.grad ** 2, dim=2))
        elif args.saliency_mode == 'gradient_input':
            unary = torch.sum(input_embeds.grad * input_embeds, dim=-1).detach()

        unary = unary / unary.view(bc, -1).max(1)[0].view(bc, 1)
        unary = unary.cpu().numpy()

        probs = torch.softmax(logits, dim=-1).cpu().detach().numpy()
        preds = logits.argmax(-1).cpu().numpy()

        for sample, label, saliency, pred, prob in zip(batch['input_ids'], batch['labels'].numpy(), unary, preds, probs):
            top_k = heapq.nlargest(K, range(len(saliency)), saliency.take) # top_k saliency
            top_k_ids = sample[top_k].numpy()
            top_k_saliency = saliency[top_k]
            flag = label == pred
            item = {"text": helper.tokenizer.decode(sample), "label": label, "pred": pred, "flag": flag, "prob": prob}
            top_k_tokens = helper.tokenizer.convert_ids_to_tokens(top_k_ids)
            item[f'top_{K}'] = [(i, token, s) for i, token, s in zip(top_k, top_k_tokens, top_k_saliency)]
            item['saliency'] = saliency
            result.append(item)
            # result.append({f"top_{K}": helper.tokenizer.convert_ids_to_tokens(top_k_ids), f"top_{K}_ids": top_k_ids.numpy(), "id": sample.numpy(), "saliency": saliency})
    labels = np.array([i['label'] for i in result])
    preds = np.array([i['pred'] for i in result])
    print("acc = ", np.mean((labels==preds)))

    with open(args.output_path, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

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

    # noise
    parser.add_argument("--noise_target", type=str, default="rationale")
    parser.add_argument("--noise_ratio", type=float, default=None)
    parser.add_argument("--noise_source", type=str, default="human-highlighted")

    # device
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of cpu cores to use')
    parser.add_argument('--gpus', default=None, help='Gpus to use')

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
    
    # model
    parser.add_argument("--model_config", type=str, default='bert-base-uncased',
                        help="config path for model(tokenizer)")
    
    # model load/save
    parser.add_argument('--load_dir', type=str, default=None,
                        help='Directory of checkpoint to load for predicting')
    parser.add_argument('--save_dir', type=str,
                        help='Path to save model')
    parser.add_argument('--output_path', type=str,
                        help='saliency analysis result')
    parser.add_argument("--predict_file", type=str, default='predict.json')

    # mode
    parser.add_argument('--predict', action='store_true',
                        help='predict result')
    parser.add_argument('--test', action='store_true',
                        help='test')
    parser.add_argument("--saliency", action='store_true',
                        help='saliency')

    # saliency
    parser.add_argument("--saliency_mode", type=str, default='gradient')

    return parser.parse_args()



if __name__ == '__main__':

    args = parse()
    for k, v in vars(args).items():
        print(f"{k}:\t{v}")

    helper = Helper(args)

    pl.seed_everything(args.seed)

    if args.predict:
        predict(args)
    elif args.test:
        test(args)
    elif args.saliency:
        if args.saliency_mode == 'erase':
            get_saliency_erase(args)
        else:
            get_saliency(args)
    else:
        train(args)