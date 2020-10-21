from datasetReader.DatasetReader import OneStageDataReader
from models.bert_for_ner import BertOneStageForNer
from transformers import BertConfig
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch import optim
from tools.finetune_argparse import get_argparse_bert_one_stage_bk
from tools.adamw import AdamW
from tqdm import tqdm
import torch
import  time
from tools.warm_up import get_linear_schedule_with_warmup
from tools.progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import random
import numpy as np






"""
这里NER采用Bert+分类器，也是直接使用一阶段的预测的方法，这里有几个细节的地方值得记录
1、优化器、学习率以及学习率衰减的设置；
2、序列数据的标注要准确，且标注的非实体类不能用0，这里要使用其他的数字别是非实体类别，因为Bert中有个padding需要用到0
3、sequence的长度在不同的Batch中要动态调整，这里的实现就是collate_fn函数，每个batch取自己的最长的那个序列作为每个sequence的长度
4、关于BertTokenizer中tokenizer.tokenize(text)的使用，数据集中含有中英文的情况，我们在做label的时候已经是安装每个汉字和每个字母做的标注
，而tokenizer.tokenize(text)则会把text中的英文安装单词和单词的一份形如——CSOL处理为‘cs’和‘##ol’，而不是‘c’,'s','o','l'。这样就会
造成这些数据的编码向量和labels向量对应不上
"""





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
    model.to(args.device)
    # train_sampler = RandomSampler(train_data)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)

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

    t_total = len(train_dataloader) * args.epochs
    args.warmup_steps = int(t_total * args.warmup_proportion)
    # optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,eps=args.adam_epsilon )

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,num_training_steps=t_total)

    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=False, threshold=0.0001,
    #                               threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # early_stop_step = 20000
    # last_improve = 0  # 记录上次提升的step
    # flag = False  # 记录是否很久没有效果提升
    model.zero_grad()
    dev_best_acc = 0
    global_step = 0

    seed_everything(args.seed)
    for epoch in tqdm(range(args.epochs), desc='Epoch'):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training Iteraction')
        total_loss = 0
        train_corrects = {}
        train_ture_totals = {}
        train_pre_totals = {}
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bio_labels': batch[2]}
            # print('input_ids',inputs['input_ids'])
            # print('input_ids',inputs['input_ids'].shape)

            outputs = model(**inputs)
            biesos_logits = outputs[1]
            # print('*' * 100)
            # print('biesos_logits', biesos_logits)
            # print('*' * 100)

            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scheduler.step()
            optimizer.step()
            model.zero_grad()
            global_step += 1
            total_loss += loss.item()
            # biesos_logits = outputs[1]
            # biesos_tags = torch.argmax(biesos_logits, dim=2).cpu().numpy().tolist()
            # bieso_labels = inputs['bio_labels'].cpu().numpy().tolist()
            #
            # ture_totals,pre_totals,corrects = compute_metrics(bieso_labels, biesos_tags,args)
            # for (t_k,t_v),(p_k,p_v),(c_k,c_v) in zip(ture_totals.items(),pre_totals.items(),corrects.items()):
            #     if t_k == c_k:
            #         if t_k not in train_ture_totals:
            #             train_ture_totals[t_k] = t_v
            #         else:
            #             train_ture_totals[t_k] += t_v
            #
            #         if p_k not in train_pre_totals:
            #             train_pre_totals[p_k] = p_v
            #         else:
            #             train_pre_totals[p_k] += p_v
            #
            #         if c_k not in train_corrects:
            #             train_corrects[c_k] = c_v
            #         else:
            #             train_corrects[c_k] += c_v
            #     else:
            #         print('train t_k != c_k')

            pbar(step, {"loss": loss.item()})
        train_loss = total_loss / len(train_dataloader)

        # train_corrects = dict(sorted(train_corrects.items(),key= lambda x:x[0]))
        # train_ture_totals = dict(sorted(train_ture_totals.items(), key=lambda x: x[0]))
        # train_pre_totals = dict(sorted(train_pre_totals.items(), key=lambda x: x[0]))
        #
        # precision,recall,f1 = get_acc(train_corrects,train_ture_totals,train_pre_totals)
        # print('='*100)
        # print('Train  corrects:',train_corrects)
        # print('*' * 100)
        # print('Train  totals:', train_ture_totals)
        # print('*' * 100)
        # print('Train    recall:',recall)
        # print('*' * 100)
        # print('Train precision:',precision)
        # print('*' * 100)
        # print('Train        f1:', f1)
        # print('*' * 100)
        # print('Train loss:{:.6f}'.format(train_loss))
        # print('\n')
        # print('\n')
        dev_loss_mean, dev_corrects, dev_precision, dev_recall, dev_f1, dev_ture_totals = evaluate(model, dev_data,
                                                                                                   args)
        # scheduler.step(dev_best_acc)
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
            biesos_tags = torch.argmax(biesos_logits, dim=2).cpu().numpy().tolist()
            # print('biesos_tags',biesos_tags)
            biesos_tags = np.argmax(biesos_logits.cpu().numpy(), axis=2).tolist()
            # print('biesos_tags', biesos_tags)
            bieso_labels = inputs['bio_labels'].cpu().numpy().tolist()
            # print('bieso_labels',bieso_labels)

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

            loss_total  += loss

    loss_mean = loss_total.item()/len(dev_dataloader)

    dev_corrects = dict(sorted(dev_corrects.items(), key=lambda x: x[0]))
    dev_ture_totals = dict(sorted(dev_ture_totals.items(), key=lambda x: x[0]))
    dev_pre_totals = dict(sorted(dev_pre_totals.items(), key=lambda x: x[0]))

    dev_precision, dev_recall, dev_f1 = get_acc(dev_corrects, dev_ture_totals, dev_pre_totals)
    return loss_mean, dev_corrects,dev_precision, dev_recall, dev_f1,dev_ture_totals

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
    # for 循环对每一条数据做统计
    for bieso_label, biesos_tag in zip(bieso_labels, biesos_tags):
        id2label = args.bios_id2label
        bieso_label = [id2label[id] for id in bieso_label]
        biesos_tag = [id2label[id] for id in biesos_tag]
        true_entities = get_entities(bieso_label)
        pre_entities = get_entities(biesos_tag)

        true_entities = dict(sorted(true_entities.items(), key=lambda x: x[0]))
        pre_entities = dict(sorted(pre_entities.items(), key=lambda x: x[0]))

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
    args = get_argparse_bert_one_stage_bk().parse_args()
    vocab_att_path = args.data_dir + '/vocab_bios_list.txt'

    with open(vocab_att_path, 'r') as f:
        bios = f.readlines()
    bios = [ele.strip('\n') for ele in bios]
    vocab_len = len(bios)
    print('vocab_len', vocab_len)
    config = BertConfig.from_pretrained(args.bert_model_path)
    config.biso_num_labels = vocab_len
    args.bios_id2label = {id: label for id, label in enumerate(bios)}
    print(args)

    print(config)
    model = BertOneStageForNer.from_pretrained(args.bert_model_path, config=config)
    train_data = OneStageDataReader(args=args, text_file_name='train_text.txt', bieso_file_name='train_bios.txt', )
    dev_data = OneStageDataReader(args=args, text_file_name='dev_text.txt', bieso_file_name='dev_bios.txt', )
    test_data = OneStageDataReader(args=args, text_file_name='test_text.txt', bieso_file_name='test_bios.txt', )
    train(model, train_data, dev_data, args)

if __name__ == '__main__':
    main()




