from datasetReader.DatasetReader import TwostageDataReader
from models.bert_for_ner import BertCrfTwoStageForNer
from transformers import BertConfig
from torch.utils.data import DataLoader,RandomSampler
from torch import optim
from tools.finetune_argparse import get_argparse_two_stage
from tqdm import tqdm
import torch
import  time
from tools.warm_up import get_linear_schedule_with_warmup


def train(model,train_data,dev_data,args):
       train_sampler = RandomSampler(train_data)
       train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
       no_decay = ["bias", "LayerNorm.weight"]

       bert_param_optimizer = list(model.bert.named_parameters())
       crf_bieso_param_optimizer = list(model.crf_bieso.named_parameters())
       crf_att_param_optimizer = list(model.crf_att.named_parameters())
       crf_param_optimizer = crf_bieso_param_optimizer+crf_att_param_optimizer

       cls_bieso_linear_param_optimizer = list(model.cls_bieso.named_parameters())
       cls_att_linear_param_optimizer = list(model.cls_att.named_parameters())
       linear_param_optimizer = cls_bieso_linear_param_optimizer+cls_att_linear_param_optimizer

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

       # early_stop_step = 20000
       # last_improve = 0  # 记录上次提升的step
       # flag = False  # 记录是否很久没有效果提升
       dev_best_acc = 0
       global_step = 0
       model.to(args.device)
       model.train()
       for epoch in tqdm(range(args.epochs),desc='Epoch'):
           total_loss = 0
           correct = 0
           total = 0
           for step,batch in  enumerate(tqdm(train_dataloader,desc='Iteraction')):
               optimizer.zero_grad()
               batch = tuple(t.to(args.device) for t in batch)
               inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bieso_labels': batch[2],'att_labels': batch[3]}
               outputs = model(**inputs)
               loss = outputs[0]
               loss.backward()
               optimizer.step()
               scheduler.step()  # Update learning rate schedule
               global_step += 1
               total_loss += loss

               biesos_logits, att_logits = outputs[1], outputs[2]
               biesos_tags, att_tags = model.crf_bieso.decode(biesos_logits,inputs['attention_mask']), model.crf_att.decode( att_logits, inputs['attention_mask'])
               biesos_tags, att_tags = biesos_tags.squeeze(0).cpu().numpy().tolist(), att_tags.squeeze(0).cpu().numpy().tolist()
               attention_masks = inputs['attention_mask'].cpu().numpy().tolist()
               bieso_labels, att_labels = inputs['bieso_labels'].cpu().numpy().tolist(), inputs['att_labels'].cpu().numpy().tolist()
               batch_total, batch_correct = compute_metrics(bieso_labels, att_labels, biesos_tags, att_tags, attention_masks)

               correct += batch_correct
               total += batch_total
               train_acc = correct/total

               lr = scheduler.get_lr()
               if global_step%64 == 0 or global_step%len(train_dataloader) == 0:
                   print('lr', lr)
                   print('Train Epoch[{}/{}],train_acc:{:.4f}%,correct/total={}/{},train_loss:{:.6f}'.format(epoch, args.epochs,train_acc*100,correct,total,loss.item()))
           # train_loss = total_loss.item()/len(train_dataloader)
           # print('Train Epoch[{}/{}]%,train_loss:{:.6f}'.format(epoch, args.epochs,train_loss))
           dev_acc, dev_loss = evaluate(model,dev_data,args)

           if dev_best_acc< dev_acc:
               dev_best_acc = dev_acc
               print('save model....')
               torch.save(model,'outputs/BertCrfTwoStageNer_model/bertCrfTwoStageNer_model.bin')
           print('Dev Acc:{:.4f}%,Best_dev_acc:{:.4f}%,dev_loss:{:.6f}'.format(dev_acc * 100, dev_best_acc * 100,dev_loss))



def evaluate(model,dev_data,args):
    dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.batch_size)
    model.eval()
    loss_total = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for step, batch in enumerate(tqdm(dev_dataloader, desc='dev iteration:')):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bieso_labels': batch[2],
                      'att_labels': batch[3]}
            outputs = model(**inputs)
            loss, biesos_logits, att_logits = outputs[0], outputs[1], outputs[2]
            biesos_tags, att_tags = model.crf_bieso.decode(biesos_logits,
                                                           inputs['attention_mask']), model.crf_att.decode(att_logits,
                                                                                                           inputs[
                                                                                                               'attention_mask'])
            biesos_tags, att_tags = biesos_tags.squeeze(0).cpu().numpy().tolist(), att_tags.squeeze(
                0).cpu().numpy().tolist()
            attention_masks = inputs['attention_mask'].cpu().numpy().tolist()
            bieso_labels, att_labels = inputs['bieso_labels'].cpu().numpy().tolist(), inputs[
                'att_labels'].cpu().numpy().tolist()

            batch_total,batch_correct = compute_metrics(bieso_labels, att_labels, biesos_tags, att_tags, attention_masks)
            correct += batch_correct
            total += batch_total
            loss_total  += loss

    loss_mean = loss_total.item()/len(dev_dataloader)
    acc = correct/total
    return acc,loss_mean

def compute_metrics(bieso_labels,att_labels,biesos_tags, att_tags,attention_masks):
       """
       这里传入的都是list，每一个list的元素是一个list，第一维list的长度是batch_size;第二维的list是max_sequence
       Args:
              bieso_labels:
              att_labels:
              biesos_tag:
              att_tag:
              attention_mask:
       Returns:
       """
       batch_total = 0
       batch_correct = 0
       #for 循环对每一条数据做统计
       for bieso_label,att_label,biesos_tag,att_tag,attention_mask in zip(bieso_labels,att_labels,biesos_tags, att_tags,attention_masks):
           seq_length = attention_mask.count(1)

           bieso_ture_entites = []#保存每条数据所有的实体[[1,2,2,3],[1,3]]
           bieso_pre_entites = []
           att_ture_entites = [] #[[7,7,7,7],[2,2,2]]
           att_pre_entites = []

           bieso_ture_entite = []#保存每个实体对应的文字index
           bieso_pre_entite = []
           att_ture_entite = []
           att_pre_entite = []


           # print('att_label',att_label[0:seq_length])
           # print('bieso_label',bieso_label[0:seq_length])

           for index in range(0, seq_length): #把att_label中所有连续的非0的字符分组统计出来；
               ture_att = att_label[index]   #以实体属性名称来确定其他的
               ture_bieso = bieso_label[index]
               pre_att = att_tag[index]
               pre_bieso = biesos_tag[index]
               if ture_att != 0:
                   if ture_att in att_ture_entite:
                       att_ture_entite.append(ture_att)
                       bieso_ture_entite.append(ture_bieso)

                       att_pre_entite.append(pre_att)
                       bieso_pre_entite.append(pre_bieso)

                   else:
                       if len(att_ture_entite) == 0:
                           att_ture_entite.append(ture_att)
                           bieso_ture_entite.append(ture_bieso)

                           att_pre_entite.append(pre_att)
                           bieso_pre_entite.append(pre_bieso)
                       else:
                           att_ture_entites.append(att_ture_entite)
                           bieso_ture_entites.append(bieso_ture_entite)
                           att_pre_entites.append(att_pre_entite)
                           bieso_pre_entites.append(bieso_pre_entite)

                           bieso_ture_entite = []
                           bieso_pre_entite = []
                           att_ture_entite = []
                           att_pre_entite = []

                           att_ture_entite.append(ture_att)
                           bieso_ture_entite.append(ture_bieso)

                           att_pre_entite.append(pre_att)
                           bieso_pre_entite.append(pre_bieso)
               else:
                   if len(att_ture_entite)>0:
                       att_ture_entites.append(att_ture_entite)
                       bieso_ture_entites.append(bieso_ture_entite)
                       att_pre_entites.append(att_pre_entite)
                       bieso_pre_entites.append(bieso_pre_entite)

                       bieso_ture_entite = []
                       bieso_pre_entite = []
                       att_ture_entite = []
                       att_pre_entite = []
               index += 1
           if len(att_ture_entite)>0:
               att_ture_entites.append(att_ture_entite)
               bieso_ture_entites.append(bieso_ture_entite)
               att_pre_entites.append(att_pre_entite)
               bieso_pre_entites.append(bieso_pre_entite)

           # print('att_ture_entites',att_ture_entites)
           # print('att_pre_entites', att_pre_entites)
           #
           # print('bieso_ture_entites', bieso_ture_entites)
           # print('bieso_pre_entites', bieso_pre_entites)


           total = len(att_ture_entites)
           correct = 0
           for att_ture_entite,att_pre_entite,bieso_ture_entite,bieso_pre_entite in zip(att_ture_entites,att_pre_entites,bieso_ture_entites,bieso_pre_entites):
               # print('att_ture_entite',att_ture_entite)
               # print('att_pre_entite',att_pre_entite)
               # print('bieso_ture_entite',bieso_ture_entite)
               # print('bieso_pre_entite',bieso_pre_entite)
               # time.sleep(5000)
               if att_ture_entite == att_pre_entite and bieso_ture_entite == bieso_pre_entite:
                   batch_correct += 1

           batch_total += total
           batch_correct += correct

       return batch_total,batch_correct






def main():
       args = get_argparse_two_stage().parse_args()
       print(args)

       vocab_att_path = args.data_dir + '/vocab_attr_list.txt'

       with open(vocab_att_path,'r') as f:
              atts = f.readlines()
       vocab_len = len(atts)
       print('vocab_len',vocab_len)
       config = BertConfig.from_pretrained(args.bert_model_path)
       config.bieso_num_labels = 5
       config.att_num_labels = vocab_len
       config.cls_dropout = 0.5

       print(config)

       model = BertCrfTwoStageForNer.from_pretrained(args.bert_model_path,config=config)

       train_data = TwostageDataReader(args=args, text_file_name='train_text.txt', bieso_file_name='train_bieso.txt',
                                       atts_file_name='train_attris.txt')
       dev_data = TwostageDataReader(args=args, text_file_name='dev_text.txt', bieso_file_name='dev_bieso.txt',
                                     atts_file_name='dev_attris.txt')
       test_data = TwostageDataReader(args=args, text_file_name='test_text.txt', bieso_file_name='test_bieso.txt',
                                      atts_file_name='test_attris.txt')

       train(model,train_data,dev_data,args)


if __name__ == '__main__':
       main()




