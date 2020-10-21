from datasetReader.DatasetReader import OneStageDataReader
from models.bert_for_ner import BertCrfOneStageForNer
from transformers import BertConfig
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch import optim
from tools.finetune_argparse import get_argparse_bert_crf_one_stage
from tqdm import tqdm
import torch
import  time
from tools.warm_up import get_linear_schedule_with_warmup
from tools.progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import os
import numpy as np



def seed_everything(seed=42):
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

def collate_fn(batch):
    """
    batch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest,
    """
    all_input_ids, all_attention_mask, all_labels,input_lens = map(torch.stack, zip(*batch))
    max_len = max(input_lens).item()
    # print('max_len',max_len)
    all_input_ids = all_input_ids[:, :max_len]
    all_attention_mask = all_attention_mask[:, :max_len]
    all_labels = all_labels[:,:max_len]
    return all_input_ids, all_attention_mask, all_labels

def train(model,train_data,dev_data,args):
       train_sampler = RandomSampler(train_data)
       train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size,collate_fn=collate_fn)
       no_decay = ["bias", "LayerNorm.weight"]

       bert_param_optimizer = model.bert.named_parameters()
       crf_param_optimizer = model.crf.named_parameters()
       linear_param_optimizer = model.cls.named_parameters()

       # print('bert_param_optimizer',len(bert_param_optimizer))
       # print('crf_bieso_param_optimizer',len(crf_bieso_param_optimizer))
       # print('crf_att_param_optimizer',len(crf_att_param_optimizer))
       # print('cls_bieso_linear_param_optimizer',len(cls_bieso_linear_param_optimizer))
       # print('cls_att_linear_param_optimizer',len(cls_att_linear_param_optimizer))
       # print('crf_param_optimizer',len(crf_param_optimizer))
       # print('linear_param_optimizer',len(linear_param_optimizer))

       optimizer_grouped_parameters = [
          {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay, 'lr': args.learning_rate},
          {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr': args.learning_rate},

          {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
          {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr': args.crf_learning_rate},

          {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
          {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,'lr': args.crf_learning_rate}
       ]

       t_total = len(train_dataloader)*args.epochs
       args.warmup_steps = int(t_total * args.warmup_proportion)
       optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,eps=args.adam_epsilon )

       scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=t_total)
       # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=False, threshold=0.0001,
       #                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

       # early_stop_step = 20000
       # last_improve = 0  # 记录上次提升的step
       # flag = False  # 记录是否很久没有效果提升
       dev_best_acc = 0
       global_step = 0
       model.to(args.device)
       model.train()
       seed_everything(args.seed)
       for epoch in tqdm(range(args.epochs),desc='Epoch'):
           pbar = ProgressBar(n_total=len(train_dataloader), desc='Training Iteraction')
           total_loss = 0
           train_corrects = {}
           train_ture_totals = {}
           train_pre_totals = {}
           for step,batch in  enumerate(train_dataloader):
               batch = tuple(t.to(args.device) for t in batch)
               inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bio_labels': batch[2]}
               outputs = model(**inputs)
               loss = outputs[0]
               loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

               optimizer.step()
               scheduler.step()
               model.zero_grad()


               global_step += 1
               total_loss += loss
               pbar(step, {"loss": loss.item()})

               biesos_logits = outputs[1]
           #     biesos_tags = model.crf.decode(biesos_logits,inputs['attention_mask'])
           #     biesos_tags = biesos_tags.squeeze(0).cpu().numpy().tolist()
           #     bieso_labels = inputs['bio_labels'].cpu().numpy().tolist()
           #
           #     # print('biesos_tags', biesos_tags)
           #     # print('*'*100)
           #     # print('bieso_labels', bieso_labels)
           #
           #     ture_totals, pre_totals, corrects = compute_metrics(bieso_labels, biesos_tags, args)
           #     for (t_k, t_v), (p_k, p_v), (c_k, c_v) in zip(ture_totals.items(), pre_totals.items(), corrects.items()):
           #         if t_k == c_k:
           #             if t_k not in train_ture_totals:
           #                 train_ture_totals[t_k] = t_v
           #             else:
           #                 train_ture_totals[t_k] += t_v
           #
           #             if p_k not in train_pre_totals:
           #                 train_pre_totals[p_k] = p_v
           #             else:
           #                 train_pre_totals[p_k] += p_v
           #
           #             if c_k not in train_corrects:
           #                 train_corrects[c_k] = c_v
           #             else:
           #                 train_corrects[c_k] += c_v
           #         else:
           #             print('train t_k != c_k')
           #     pbar(step, {"loss": loss.item()})
           # train_loss = total_loss / len(train_dataloader)
           # train_corrects = dict(sorted(train_corrects.items(), key=lambda x: x[0]))
           # train_ture_totals = dict(sorted(train_ture_totals.items(), key=lambda x: x[0]))
           # train_pre_totals = dict(sorted(train_pre_totals.items(), key=lambda x: x[0]))
           #
           # precision, recall, f1 = get_acc(train_corrects, train_ture_totals, train_pre_totals)
           # print('=' * 100)
           # print('Train  corrects:', train_corrects)
           # print('*' * 100)
           # print('Train  totals:', train_ture_totals)
           # print('*' * 100)
           # print('Train    recall:', recall)
           # print('*' * 100)
           # print('Train precision:', precision)
           # print('*' * 100)
           # print('Train        f1:', f1)
           # print('*' * 100)
           # print('Train loss:{:.6f}'.format(train_loss))
           # print('\n')
           # print('\n')
           #
           # # train_acc = correct / total
           # # bert_lr = optimizer.param_groups[0]['lr']
           # # train_loss = total_loss/ len(train_dataloader)
           # # print(' Train Epoch[{}/{}],train_acc:{:.4f}%,correct/total={}/{},train_loss:{:.6f},bert_lr:{}'.format(epoch,args.epochs,train_acc * 100,correct, total,train_loss,bert_lr))

           dev_loss_mean, dev_corrects, dev_precision, dev_recall, dev_f1, dev_ture_totals = evaluate(model, dev_data,
                                                                                                      args)
           scheduler.step(dev_best_acc)
           if dev_best_acc < dev_recall['name']:
               dev_best_acc = dev_recall['name']
               print('save model....')
               torch.save(model, 'outputs/BertOneStageNer_model/BertOneStageNer_model.bin')
           print('=' * 100)
           print('Dev  corrects:', dev_corrects)
           print('*' * 100)
           print('Dev    totals:', dev_ture_totals)
           print('*' * 100)
           print('Dev    recall:', dev_recall)
           print('*' * 100)
           print('Dev precision:', dev_precision)
           print('*' * 100)
           print('Dev        f1:', dev_f1)
           print('=' * 100)
           print('Best_dev_acc:{:.4f}%,dev_loss:{:.6f}'.format(dev_best_acc * 100, dev_loss_mean))
           print('\n')
           print('\n')

def get_acc(corrects,true_totals,pre_totals):
    precision = {}
    recall = {}
    f1 = {}
    for (t_k,t_v),(p_k,p_v),(c_k,c_v) in zip(true_totals.items(),pre_totals.items(),corrects.items()):
        if t_k == c_k:
            if t_v > 0:
                recall[t_k] = c_v/t_v
            else:
                recall[t_k] = 0
            if p_v > 0:
                precision[p_k] = c_v/p_v
            else:
                precision[p_k] = 0
            if (t_v+p_v) != 0:
                f1[p_k] = 2*c_v/(t_v+p_v)
            else:
                f1[p_k] = 0

    precision = dict(sorted(precision.items(), key= lambda x:x[0]))
    recall = dict(sorted(recall.items(), key=lambda x: x[0]))
    f1 = dict(sorted(f1.items(), key=lambda x: x[0]))
    return precision,recall,f1




def evaluate(model,dev_data,args):
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.batch_size,collate_fn=collate_fn)
    model.eval()
    loss_total = 0
    with torch.no_grad():
        dev_corrects = {}
        dev_ture_totals = {}
        dev_pre_totals = {}
        for step, batch in enumerate(tqdm(dev_dataloader, desc='dev iteration:')):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bio_labels': batch[2],}
            outputs = model(**inputs)
            loss, biesos_logits = outputs[0], outputs[1]
            biesos_tags = model.crf.decode(biesos_logits,inputs['attention_mask'])
            biesos_tags = biesos_tags.squeeze(0).cpu().numpy().tolist()
            bieso_labels = inputs['bio_labels'].cpu().numpy().tolist()

            # print('biesos_tags', biesos_tags)
            # print('*' * 100)
            # print('bieso_labels', bieso_labels)

            ture_totals, pre_totals, corrects = compute_metrics(bieso_labels, biesos_tags, args)
            for (t_k, t_v), (p_k, p_v), (c_k, c_v) in zip(ture_totals.items(), pre_totals.items(), corrects.items()):
                if t_k == c_k:
                    if t_k not in dev_ture_totals:
                        dev_ture_totals[t_k] = t_v
                    else:
                        dev_ture_totals[t_k] += t_v

                    if p_k not in dev_pre_totals:
                        dev_pre_totals[p_k] = p_v
                    else:
                        dev_pre_totals[p_k] += p_v

                    if c_k not in dev_corrects:
                        dev_corrects[c_k] = c_v
                    else:
                        dev_corrects[c_k] += c_v
                else:
                    print('train t_k != c_k')

            loss_total += loss

        loss_mean = loss_total.item() / len(dev_dataloader)

        dev_corrects = dict(sorted(dev_corrects.items(), key=lambda x: x[0]))
        dev_ture_totals = dict(sorted(dev_ture_totals.items(), key=lambda x: x[0]))
        dev_pre_totals = dict(sorted(dev_pre_totals.items(), key=lambda x: x[0]))

        dev_precision, dev_recall, dev_f1 = get_acc(dev_corrects, dev_ture_totals, dev_pre_totals)
        return loss_mean, dev_corrects, dev_precision, dev_recall, dev_f1, dev_ture_totals


def compute_metrics(bieso_labels,biesos_tags,args):
       """
       这里传入的都是list，每一个list的元素是一个list，第一维list的长度是batch_size;第二维的list是max_sequence
       Args:
           bieso_labels:
           biesos_tags:
       Returns:
       """
       ture_totals = {}  # TP_FN 计算recall的
       pre_totals = {}  # TP_FP 计算precision的
       corrects = {}
       #for 循环对每一条数据做统计
       for bieso_label,biesos_tag in zip(bieso_labels,biesos_tags):
           id2label = args.bios_id2label
           bieso_label = [id2label[id] for id in bieso_label]
           biesos_tag = [id2label[id] for id in biesos_tag]
           # print('bieso_label',bieso_label)
           # print('biesos_tag',biesos_tag)
           # time.sleep(5000)
           true_entities = get_entities(bieso_label)
           pre_entities = get_entities(biesos_tag)

           for (ture_k, ture_v), (pre_k, pre_v) in zip(true_entities.items(), pre_entities.items()):
               if ture_k == pre_k:
                   pre_total = len(pre_v)
                   ture_total = len(ture_v)
                   correct = len([ele for ele in pre_v if ele in ture_v])

                   if ture_k not in ture_totals:
                       ture_totals[ture_k] = ture_total
                   else:
                       ture_totals[ture_k] += ture_total

                   if ture_k not in pre_totals:
                       pre_totals[ture_k] = pre_total
                   else:
                       pre_totals[ture_k] += pre_total

                   if ture_k not in corrects:
                       corrects[ture_k] = correct
                   else:
                       corrects[ture_k] += correct
               else:
                   print('compute_metrics ture_k != pre_k ')

       ture_totals = dict(sorted(ture_totals.items(), key=lambda x: x[0]))
       pre_totals = dict(sorted(pre_totals.items(), key=lambda x: x[0]))
       corrects = dict(sorted(corrects.items(), key=lambda x: x[0]))
       return ture_totals, pre_totals, corrects



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


    per_names = []
    books = []
    addresses = []
    companys = []
    games = []
    governments = []
    movies = []
    organizations = []
    positions = []
    scenes = []

    for entity in chunks:
        if 'name' == entity[0]:
            per_names.append(entity)
        if 'book' == entity[0]:
            books.append(entity)
        if 'address' == entity[0]:
            addresses.append(entity)
        if 'company' == entity[0]:
            companys.append(entity)
        if 'game' == entity[0]:
            games.append(entity)
        if 'government' == entity[0]:
            governments.append(entity)
        if 'movie' == entity[0]:
            movies.append(entity)
        if 'organization' == entity[0]:
            organizations.append(entity)
        if 'position' == entity[0]:
            positions.append(entity)
        if 'scene' == entity[0]:
            scenes.append(entity)
    entitys = {}
    entitys['name'] = per_names
    entitys['book'] = books
    entitys['address'] = addresses
    entitys['company'] = companys
    entitys['government'] = governments
    entitys['movie'] = movies
    entitys['organization'] = organizations
    entitys['position'] = positions
    entitys['scene'] = scenes
    entitys['game'] = games

    entitys = dict(sorted(entitys.items(),key= lambda x:x[0]))

    # print('per_names',per_names)
    return entitys


def main():
    args = get_argparse_bert_crf_one_stage().parse_args()
    vocab_att_path = args.data_dir + '/vocab_bios_list.txt'

    with open(vocab_att_path, 'r') as f:
        bios = f.readlines()
    bios = [ele.strip('\n') for ele in bios]
    vocab_len = len(bios)
    print('vocab_len', vocab_len)
    config = BertConfig.from_pretrained(args.bert_model_path)
    config.biso_num_labels = vocab_len
    # config.cls_dropout = 0.5
    args.bios_id2label = {id: label for id, label in enumerate(bios)}
    print(args)
    print(config)
    model = BertCrfOneStageForNer.from_pretrained(args.bert_model_path, config=config)
    train_data = OneStageDataReader(args=args, text_file_name='train_text.txt', bieso_file_name='train_bios.txt', )
    dev_data = OneStageDataReader(args=args, text_file_name='dev_text.txt', bieso_file_name='dev_bios.txt', )
    test_data = OneStageDataReader(args=args, text_file_name='test_text.txt', bieso_file_name='test_bios.txt', )

    train(model, train_data, dev_data, args)



if __name__ == '__main__':
       main()




