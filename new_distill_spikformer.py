import torch
import os
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
from dataset import TxtDataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
from spikingjelly.activation_based import encoding
from spikingjelly.activation_based import functional
import math
from utils.public import set_seed
from torchmetrics.classification import MatthewsCorrCoef

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0,2,3"

def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)
    

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset_name",default="sst2",type=str)
    parser.add_argument("--data_augment", default="True", type=str)
    parser.add_argument("--batch_size",default=4,type=int)
    parser.add_argument("--fine_tune_lr",default=1e-2,type=float)
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--teacher_model_path",default="", type=str)
    parser.add_argument("--label_num",default=2,type=int)
    parser.add_argument("--depths",default=6,type=int)
    parser.add_argument("--max_length",default=64,type=int)
    parser.add_argument("--dim",default=768,type=int)
    parser.add_argument("--ce_weight", default=0.0, type=float)
    parser.add_argument("--emb_weight", default=1.0, type=float)
    parser.add_argument("--logit_weight", default=1.0, type=float)
    parser.add_argument("--rep_weight", default=5.0, type=float)
    parser.add_argument("--num_step", default=32, type=int)
    parser.add_argument("--tau", default=10.0, type=float)
    parser.add_argument("--common_thr", default=1.0, type=float)
    parser.add_argument("--predistill_model_path", default="", type=str)
    # parser.add_argument("--predistill_requires_grad", default="True", type=str)
    parser.add_argument("--ignored_layers", default=1, type=int)
    parser.add_argument("--metric", default="acc", type=str)
    args = parser.parse_args()
    return args

def distill(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path)
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model_path, num_labels=args.label_num, output_hidden_states=True).to(device)
    
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    student_model = new_spikformer(depths=args.depths, length=args.max_length, T=args.num_step, \
        tau=args.tau, common_thr=args.common_thr, vocab_size = len(tokenizer), dim=args.dim, num_classes=args.label_num, mode="distill")
    
    # load embedding layer
    # student_model.emb.weight = teacher_model.bert.embeddings.word_embeddings.weight
    # student_model.emb.weight.requires_grad = True
    
    if args.predistill_model_path != "":
        weights = torch.load(args.predistill_model_path)
        # for key in weights.keys():
        #     if "module.transforms" not in key:
        #         weights[key] = weights[key].float()
        #         if args.predistill_requires_grad == "False":
        #             weights[key].requires_grad = False
        #         elif args.predistill_requires_grad == "True":
        #             weights[key].requires_grad = True
        student_model.load_state_dict(weights, strict=False)
        print("load predistill model finish!")
        
    
    scaler = torch.cuda.amp.GradScaler()
    optimer = torch.optim.AdamW(params=student_model.parameters(), lr=args.fine_tune_lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimer, T_max=args.epochs, eta_min=0)
    # scheduler_warmup = GradualWarmupScheduler(optimer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    
    if args.data_augment == "True":
        print("With Augmentation")
        train_dataset = TxtDataset(data_path=f"data/{args.dataset_name}/train_augment.txt")
    else:
        print("Without Augmentation")
        train_dataset = TxtDataset(data_path=f"data/{args.dataset_name}/train.txt")
    train_data_loader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, drop_last=False)
    
    test_dataset = TxtDataset(data_path=f"data/{args.dataset_name}/test.txt")
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False)
    
    valid_dataset = TxtDataset(data_path=f"data/{args.dataset_name}/validation.txt")
    valid_data_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,drop_last=False)
    
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(device_ids)
    if len(device_ids) > 1:
        student_model = nn.DataParallel(student_model, device_ids=device_ids).to(device)
    student_model = student_model.to(device)

    metric_list = []
    for epoch in tqdm(range(args.epochs)):
        # if epoch == 5:
        #     args.rep_weight == 0
        total_loss_list = []
        embeddings_loss_list = []
        ce_loss_list = []
        logit_loss_list = []
        rep_loss_list = []
        for batch in tqdm(train_data_loader):
            student_model.train()
            batch_size = len(batch[0])
            labels = batch[1].to(device)
            inputs = tokenizer(batch[0], padding="max_length", truncation=True, \
                return_tensors="pt", max_length=args.max_length)
            # inputs = tokenizer(batch[0], padding=True, truncation=True, \
            #     return_tensors="pt", max_length=args.max_length)
            
            to_device(inputs, device)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
            # [1:] means 12 outputs of each layer; [::x] means get value every x layers 
            tea_embeddings = teacher_model.bert.embeddings.word_embeddings.weight
            if len(device_ids) > 1:
                stu_embeddings = student_model.module.emb.weight
            else:
                stu_embeddings = student_model.emb.weight

            embeddings_loss = F.mse_loss(stu_embeddings, tea_embeddings)
            embeddings_loss_list.append(embeddings_loss.item())


            tea_rep = teacher_outputs.hidden_states[1:][::int(12/args.depths)] # layers output
            # len(stu_rep) = depth
            # stu_rep[0] shape: B L D
            stu_rep, student_outputs = student_model(inputs['input_ids'])

            student_outputs = student_outputs.reshape(-1 , args.num_step, args.label_num)

            # Before transpose: B T Label_num
            # After transpose:  T B Label_num
            student_outputs = student_outputs.transpose(0, 1)

            student_logits = torch.mean(student_outputs, dim=0) # B Label_num
            # last step
            # student_logits =  student_outputs[-1,:,:]# B Label_num

            
            # print("student_logits.shape: ", student_logits.shape)
            # print("labels.shape: ", labels.shape)
            ce_loss = F.cross_entropy(student_logits, labels)
            ce_loss_list.append(ce_loss.item())
            # print("ce_loss: ", ce_loss, ce_loss.dtype)

            # print("student_logits.shape: ", student_logits.shape)
            # print("teacher_outputs.logits.shape: ", teacher_outputs.logits.shape)
            logit_loss = F.kl_div(F.log_softmax(student_logits, dim=1), F.softmax(teacher_outputs.logits, dim=1), reduction='batchmean')
            logit_loss_list.append(logit_loss.item())
            # print("logit_loss: ", logit_loss, logit_loss.dtype)

            tea_rep = torch.tensor(np.array([item.cpu().detach().numpy() for item in tea_rep]), dtype=torch.float32)
            tea_rep = tea_rep.to(device=device)
            
            rep_loss = 0
            tea_rep = tea_rep[args.ignored_layers:]
            stu_rep = stu_rep[args.ignored_layers:]
            # print(len(stu_rep))
            for i in range(len(stu_rep)):
                # print("stu_rep[i]", stu_rep[i])
                # print("tea_rep[i]", tea_rep[i])
                rep_loss += F.mse_loss(stu_rep[i], tea_rep[i])
            rep_loss = rep_loss / batch_size # batch mean
            rep_loss_list.append(rep_loss.item())
            # print("rep_loss: ",rep_loss)

            total_loss = (args.emb_weight * embeddings_loss) \
                + (args.ce_weight * ce_loss) \
                + (args.logit_weight * logit_loss) \
                + (args.rep_weight * rep_loss)
            # print("total_loss: ", total_loss.item())
            # print("total_loss:{} | ce_loss {} | logit_loss {} | rep_loss {} ".format(total_loss.item(), \
            #     ce_loss.item(), logit_loss.item(), rep_loss))
            total_loss_list.append(total_loss.item())

            optimer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimer)
            scaler.update()
            functional.reset_net(student_model)
        
            # print(
            #     f"In average, at epoch {epoch}, " 
            #     + f"ce_loss: {np.mean(ce_loss_list)}, "
            #     + f"emb_loss: {np.mean(embeddings_loss_list)} "
            #     + f"logit_loss: {np.mean(logit_loss_list)}, " 
            #     + f"rep_loss: {np.mean(rep_loss_list)}, " 
            #     + f"total_loss: {np.mean(total_loss_list)}"
            # )

        scheduler.step()
        # scheduler_warmup.step()

        y_true = []
        y_pred = []
        student_model.eval()
        with torch.no_grad():
            for batch in tqdm(test_data_loader):
            # for batch in tqdm(valid_data_loader):
                batch_size = len(batch[0])
                b_y = batch[1]
                y_true.extend(b_y.to("cpu").tolist())
                inputs = tokenizer(batch[0], padding="max_length", truncation=True, \
                        return_tensors='pt', max_length=args.max_length)
                # inputs = tokenizer(batch[0], padding=True, truncation=True, \
                #     return_tensors="pt", max_length=args.max_length)
                to_device(inputs, device)
                _, outputs = student_model(inputs['input_ids'])
                outputs = outputs.to("cpu")
                outputs = outputs.reshape(-1, args.num_step, args.label_num)
                # print("outputs.shape", outputs.shape)
                # Before transpose: B T Label_num
                # After transpose:  T B Label_num
                outputs = outputs.transpose(0, 1)
                # print("After transpose, outputs.shape", outputs.shape)
                logits = torch.mean(outputs, dim=0) # B Label_num
                
                # print("logits.shape", logits.shape)
                y_pred.extend(torch.max(logits,1)[1].tolist())

                # for line in torch.softmax(logits, 1):
                #     result.append(line.numpy())
                # correct += int(b_y.eq(torch.max(logits,1)[1]).sum())

                functional.reset_net(student_model)

        if args.metric == "acc":
            correct = 0
            for i in range(len(y_true)):
                correct += 1 if y_true[i] == y_pred[i] else 0
            acc = correct / len(y_pred)
            print("acc", acc)
        elif args.metric == "mcc":
            print(y_true)
            print(y_pred)
            matthews_corrcoef = MatthewsCorrCoef(task='binary')
            mcc = matthews_corrcoef(torch.tensor(y_true), torch.tensor(y_pred))
            print("mcc, ", mcc)

        record = acc if args.metric == "acc" else mcc
        metric_list.append(record)
        print(f"Epoch {epoch} {args.metric}: {record}")
        if record >= np.max(metric_list):
            torch.save(student_model.state_dict(),
                    f"saved_models/distilled_spikformer/{args.dataset_name}_epoch{epoch}_{record}" + 
                    f"num_step_{args.num_step}_lr{args.fine_tune_lr}_seed{args.seed}" + 
                    f"_batch_size{args.batch_size}_depths{args.depths}_max_length{args.max_length}" + 
                    f"_ce_weight{args.ce_weight}_logit_weight{args.logit_weight}_rep_weight{args.rep_weight}" +
                    f"_tau{args.tau}_common_thr{args.common_thr}"
                    )
    print("best: ", np.max(metric_list))
    return


if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(arg, getattr(_args, arg))
    set_seed(_args.seed)
    distill(_args)