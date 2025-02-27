from datasetReader.DatasetReader import OneStageDataReader
from models.bert_for_ner import BertOneStageForNer
from transformers import BertConfig
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler,TensorDataset
from torch import optim
from tools.finetune_argparse import get_argparse_bert_one_stage
from tools.adamw import AdamW
from tqdm import tqdm
import torch
import  time
from tools.warm_up import get_linear_schedule_with_warmup
from tools.progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau

from processors.utils_ner import get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
import logging
from transformers import BertTokenizer


import os
import random
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



# def collate_fn(batch):
#     """
#     batch should be a list of (sequence, target, length) tuples...
#     Returns a padded tensor of sequences sorted from longest to shortest,
#     """
#     all_input_ids, all_attention_mask, all_labels,input_lens = map(torch.stack, zip(*batch))
#     max_len = max(input_lens).item()
#     # print('max_len',max_len)
#     all_input_ids = all_input_ids[:, :max_len]
#     all_attention_mask = all_attention_mask[:, :max_len]
#     all_labels = all_labels[:,:max_len]
#     return all_input_ids, all_attention_mask, all_labels


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
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bio_labels': batch[3]}
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
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bio_labels': batch[3],}
            outputs = model(**inputs)
            loss, biesos_logits = outputs[0], outputs[1]
            # biesos_tags = torch.argmax(biesos_logits, dim=2).cpu().numpy().tolist()
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
        id2label = args.id2label
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
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk =  [-1, -1, -1]
            chunk[1] = index
            chunk[2] = index
            chunk[0] = tag.split('_')[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = index
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
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


def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_soft-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.bert_model_path.split('/'))).pop(),
        str(args.train_max_sentence_length if data_type=='train' else args.dev_max_sentence_length),
        str(task)))
    if os.path.exists(cached_features_file) :
        logging.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logging.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_sentence_length if data_type=='train' \
                                                               else args.dev_max_sentence_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token = tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logging.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_lens,all_label_ids)
    return dataset

def main():
    args = get_argparse_bert_one_stage().parse_args()
    args.task_name = 'myner'
    args.local_rank = -1
    args.model_type = 'bert'

    config = BertConfig.from_pretrained(args.bert_model_path)
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    config.biso_num_labels = num_labels


    print(config)
    model = BertOneStageForNer.from_pretrained(args.bert_model_path, config=config)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)



    # train_data = OneStageDataReader(args=args, text_file_name='train_text.txt', bieso_file_name='train_bios.txt', )
    # dev_data = OneStageDataReader(args=args, text_file_name='dev_text.txt', bieso_file_name='dev_bios.txt', )
    # test_data = OneStageDataReader(args=args, text_file_name='test_text.txt', bieso_file_name='test_bios.txt', )


    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, data_type='train')

    dev_data = load_and_cache_examples(args, args.task_name, tokenizer, data_type='dev')

    train(model, train_dataset, dev_data, args)

if __name__ == '__main__':
    main()




