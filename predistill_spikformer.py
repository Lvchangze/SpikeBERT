import torch
import torch.nn as nn
import pickle
import argparse
import torch.nn.functional as F
import torch.optim as optim
from model import new_spikformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
from transformers import BertTokenizer, BertModel
# from spikingjelly.activation_based import encoding
from spikingjelly.activation_based import functional
import math
from utils.public import set_seed
from dataset import TxtDataset, TextDataset, ChnWikiDataset
from datasets import concatenate_datasets, load_dataset
import random

print(torch.__version__)

def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--batch_size",default=32,type=int)
    parser.add_argument("--fine_tune_lr",default=6e-4,type=float)
    parser.add_argument("--max_sample_num", default=2e7, type=int)
    parser.add_argument("--epochs",default=1,type=int)
    parser.add_argument("--label_num",default=2,type=int)
    parser.add_argument("--depths",default=12,type=int)
    parser.add_argument("--max_length",default=256,type=int)
    parser.add_argument("--dim",default=768,type=int)
    # parser.add_argument("--logit_weight", default=1.0, type=float)
    parser.add_argument("--rep_weight", default=0.1, type=float)
    parser.add_argument("--tau", default=10.0, type=float)
    parser.add_argument("--common_thr", default=1.0, type=float)
    parser.add_argument("--num_step", default=16, type=int)
    parser.add_argument("--teacher_model_path", default="bert-base-cased", type=str)
    parser.add_argument("--ignored_layers", default=0, type=int)
    args = parser.parse_args()
    return args

def train(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path)
    teacher_model = BertModel.from_pretrained(args.teacher_model_path, num_labels=args.label_num, output_hidden_states=True).to(device)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    student_model = new_spikformer(depths=args.depths, length=args.max_length, T=args.num_step, \
        tau=args.tau, common_thr=args.common_thr, vocab_size = len(tokenizer), dim=args.dim, num_classes=args.label_num, mode="pre_distill")
    
    scaler = torch.cuda.amp.GradScaler()
    optimer = torch.optim.AdamW(params=student_model.parameters(), lr=args.fine_tune_lr)

    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

    # # assert bookcorpus.features.type == wiki.features.type
    raw_dataset = concatenate_datasets([bookcorpus, wiki])
    text_dataset = TextDataset(raw_dataset=raw_dataset)
    # print(text_dataset[1])
    # text_dataset = ChnWikiDataset(data_path="data/wiki_zh_2019.txt")
    
    train_data_loader = DataLoader(dataset=text_dataset, \
        batch_size= args.batch_size, shuffle=True, drop_last=False)

    # train_dataset = TxtDataset(data_path=f"data/mr/train.txt")
    # train_data_loader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, drop_last=False)
    

    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if len(device_ids) > 1:
        student_model = nn.DataParallel(student_model, device_ids=device_ids).to(device)
    student_model = student_model.to(device)

    train_iter = 0
    skip_p = args.max_sample_num / text_dataset.__len__()
    print(f"all samples:{text_dataset.__len__()}, skip_p:{skip_p}")
    for batch in tqdm(train_data_loader):
        p = random.uniform(0,1)
        if p > skip_p:
            continue
        train_iter += 1
        batch_size = len(batch)
        student_model.train()
        inputs = tokenizer(batch, padding="max_length", truncation=True, \
            return_tensors="pt", max_length=args.max_length)
        to_device(inputs, device)
        with torch.no_grad():
            teacher_outputs = teacher_model(**inputs)
        # [1:] means 12 outputs of each layer; [::x] means get value every x layers 
        tea_rep = teacher_outputs.hidden_states[1:][::int(12/args.depths)] # layers output
        # len(stu_rep) = depth
        stu_rep, student_outputs = student_model(inputs['input_ids'])

        student_outputs = student_outputs.reshape(-1 , args.num_step, args.label_num)

        # Before transpose: B T Label_num
        # After transpose:  T B Label_num
        student_outputs = student_outputs.transpose(0, 1) 
        # student_logits = torch.mean(student_outputs, dim=0) # B Label_num
        # print("student_logits.shape: ", student_logits.shape)

        # print("student_logits.shape: ", student_logits.shape)
        # print("teacher_outputs.logits.shape: ", teacher_outputs.logits.shape)
        # logit_loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_outputs.logits, dim=1), reduction='batchmean')

        tea_rep = torch.tensor(np.array([item.cpu().detach().numpy() for item in tea_rep]), dtype=torch.float32)
        tea_rep = tea_rep.to(device=device)
        
        rep_loss = 0
        tea_rep = tea_rep[args.ignored_layers:]
        stu_rep = stu_rep[args.ignored_layers:]
        for i in range(len(stu_rep)):
            rep_loss += F.mse_loss(stu_rep[i], tea_rep[i])
        rep_loss = rep_loss / batch_size # batch mean

        # total_loss = (args.logit_weight * logit_loss) +  (args.rep_weight * rep_loss)
        total_loss = args.rep_weight * rep_loss

        optimer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.step(optimer)
        scaler.update()
        functional.reset_net(student_model)

        print(f"In iter {train_iter}, "
            # + f"logit_loss: {logit_loss}, " 
            + f"rep_loss: {rep_loss}, " 
            + f"total_loss: {total_loss}"
        )

    torch.save(student_model.state_dict(), \
        f"saved_models/predistill_spikformer/" + f"_lr{args.fine_tune_lr}_seed{args.seed}" + 
        f"_batch_size{args.batch_size}_depths{args.depths}_max_length{args.max_length}" + 
        f"_tau{args.tau}_common_thr{args.common_thr}"
    )  
    return


if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    train(_args)