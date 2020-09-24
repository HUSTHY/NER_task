from datasetReader.DatasetReader import OneStageDataReader
from models.bert_for_ner import BertOneStageForNer
from transformers import BertConfig
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch import optim
from tools.finetune_argparse import get_argparse_bert_one_stage
from tqdm import tqdm
import torch
import  time
from tools.warm_up import get_linear_schedule_with_warmup
from tools.progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import random
import numpy as np

def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True



def train(model,train_data,dev_data,args):
       train_sampler = RandomSampler(train_data)
       train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)


       # print('bert_param_optimizer',len(bert_param_optimizer))
       # print('crf_bieso_param_optimizer',len(crf_bieso_param_optimizer))
       # print('crf_att_param_optimizer',len(crf_att_param_optimizer))
       # print('cls_bieso_linear_param_optimizer',len(cls_bieso_linear_param_optimizer))
       # print('cls_att_linear_param_optimizer',len(cls_att_linear_param_optimizer))
       # print('crf_param_optimizer',len(crf_param_optimizer))
       # print('linear_param_optimizer',len(linear_param_optimizer))


       no_decay = ["bias", "LayerNorm.weight"]
       optimizer_grouped_parameters = [
           {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay, },
           {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
       ]

       t_total = len(train_dataloader)*args.epochs
       args.warmup_steps = int(t_total * args.warmup_proportion)
       optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,eps=args.adam_epsilon )


       scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=t_total)

       scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False, threshold=0.0001,
                                     threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

       # early_stop_step = 20000
       # last_improve = 0  # 记录上次提升的step
       # flag = False  # 记录是否很久没有效果提升
       model.zero_grad()
       dev_best_acc = 0
       global_step = 0
       model.to(args.device)
       model.train()
       for epoch in tqdm(range(args.epochs),desc='Epoch'):
           pbar = ProgressBar(n_total=len(train_dataloader), desc='Training Iteraction')
           total_loss = 0
           correct = 0
           total = 0
           for step,batch in  enumerate(train_dataloader):
               batch = tuple(t.to(args.device) for t in batch)
               inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bio_labels': batch[2]}
               outputs = model(**inputs)
               loss = outputs[0]
               loss.backward()
               optimizer.step()
               model.zero_grad()
               global_step += 1
               total_loss += loss.item()
               biesos_logits = outputs[1]
               biesos_tags = torch.argmax(biesos_logits, dim=2).cpu().numpy().tolist()
               bieso_labels = inputs['bio_labels'].cpu().numpy().tolist()
               # if step%300 == 0 and step !=0:
               #     print('biesos_tags', biesos_tags)
               #     print('bieso_labels', bieso_labels)
               batch_total, batch_correct = compute_metrics(bieso_labels, biesos_tags,args)
               correct += batch_correct
               total += batch_total

               pbar(step,{"loss":loss.item()})
               if total > 0:
                train_acc = correct/total
               else:
                   train_acc = 0
               # if global_step%8 == 0 :
               #     bert_lr = optimizer.param_groups[0]['lr']
               #     print(' Train Epoch[{}/{}],train_acc:{:.4f}%,correct/total={}/{},train_loss:{:.6f},bert_lr:{}'.format( epoch, args.epochs, train_acc * 100, correct, total, loss.item(), bert_lr))
           train_loss = total_loss/len(train_dataloader)
           print(' Train Epoch[{}/{}],train_acc:{:.4f}%,correct/total={}/{},train_loss:{:.6f}'.format(epoch,args.epochs,train_acc * 100, correct, total,loss.item()))



           # train_acc = correct / total
           # bert_lr = optimizer.param_groups[0]['lr']
           # train_loss = total_loss/ len(train_dataloader)
           # print(' Train Epoch[{}/{}],train_acc:{:.4f}%,correct/total={}/{},train_loss:{:.6f},bert_lr:{}'.format(epoch,args.epochs,train_acc * 100,correct, total,train_loss,bert_lr))
           dev_acc, dev_loss,dev_correct,dev_total = evaluate(model,dev_data,args)
           scheduler.step(dev_best_acc)
           if dev_best_acc< dev_acc:
               dev_best_acc = dev_acc
               print('save model....')
               torch.save(model,'outputs/BertOneStageNer_model/BertOneStageNer_model.bin')
           print(' Dev Acc:{:.4f}%,correct/total={}/{},Best_dev_acc:{:.4f}%,dev_loss:{:.6f}'.format(dev_acc * 100,dev_correct,dev_total,dev_best_acc * 100,dev_loss))





def evaluate(model,dev_data,args):
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.batch_size)
    model.eval()
    loss_total = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for step, batch in enumerate(tqdm(dev_dataloader, desc='dev iteration:')):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bio_labels': batch[2],}
            outputs = model(**inputs)
            loss, biesos_logits = outputs[0], outputs[1]
            biesos_tags = torch.argmax(biesos_logits, dim=2).cpu().numpy().tolist()
            # print('biesos_tags',biesos_tags)
            biesos_tags = np.argmax(biesos_logits.cpu().numpy(), axis=2).tolist()
            # print('biesos_tags', biesos_tags)
            bieso_labels = inputs['bio_labels'].cpu().numpy().tolist()
            # print('bieso_labels',bieso_labels)

            batch_total,batch_correct = compute_metrics(bieso_labels, biesos_tags,args)
            correct += batch_correct
            total += batch_total
            loss_total  += loss

    loss_mean = loss_total.item()/len(dev_dataloader)
    if total >0:
        acc = correct/total
    else:
        acc = 0
    return acc,loss_mean,correct,total

def compute_metrics(bieso_labels,biesos_tags,args):
       """
       这里传入的都是list，每一个list的元素是一个list，第一维list的长度是batch_size;第二维的list是max_sequence
       Args:
           bieso_labels:
           biesos_tags:
       Returns:
       """
       batch_total = 0
       batch_correct = 0
       #for 循环对每一条数据做统计
       for bieso_label,biesos_tag in zip(bieso_labels,biesos_tags):
           id2label = args.bios_id2label
           bieso_label = [id2label[id] for id in bieso_label]
           biesos_tag = [id2label[id] for id in biesos_tag]
           # print('bieso_label',bieso_label)
           # print('biesos_tag',biesos_tag)
           true_entities = get_entities(bieso_label)
           # print('*'*100)
           pre_entities = get_entities(biesos_tag)
           total = len(true_entities)
           rights_entities = [pre_entitie for pre_entitie in pre_entities if pre_entitie in true_entities]
           correct = len(rights_entities)
           batch_total += total
           batch_correct += correct

       return batch_total,batch_correct



def get_entities(bieso_label):
    # print('bieso_label',bieso_label)
    chunks = []
    chunk = [-1, -1, -1]
    for index,tag in enumerate(bieso_label):
        if tag.startswith("S_"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk =  [-1, -1, -1]
            chunk[1] = index
            chunk[2] = index
            chunk[0] = tag.split('_')[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        if tag.startswith("B_"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[0] = tag.split('_')[1]
        elif tag.startswith('I_') and chunk[1] != -1:
            _type = tag.split('_')[1]
            if _type == chunk[0]:
                chunk[2] = index
            if index == len(bieso_label)-1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    # print('chunks',chunks)
    per_names = []

    for entity in chunks:
        if 'name' == entity[0]:
            per_names.append(entity)

    # print('per_names',per_names)

    return per_names


def main():
       args = get_argparse_bert_one_stage().parse_args()
       vocab_att_path = args.data_dir + '/vocab_bios_list.txt'

       with open(vocab_att_path,'r') as f:
              bios = f.readlines()
       bios = [ele.strip('\n') for ele in bios]
       vocab_len = len(bios)
       print('vocab_len',vocab_len)
       config = BertConfig.from_pretrained(args.bert_model_path)
       config.biso_num_labels = vocab_len
       args.bios_id2label = { id:label for id,label in enumerate(bios)}
       print(args)

       seed_everything(args.seed)

       print(config)
       model = BertOneStageForNer.from_pretrained(args.bert_model_path,config=config)
       train_data = OneStageDataReader(args=args, text_file_name='train_text.txt', bieso_file_name='train_bios.txt',)
       dev_data = OneStageDataReader(args=args, text_file_name='dev_text.txt', bieso_file_name='dev_bios.txt',)
       test_data = OneStageDataReader(args=args, text_file_name='test_text.txt', bieso_file_name='test_bios.txt',)

       train(model,train_data,dev_data,args)


if __name__ == '__main__':
       main()




