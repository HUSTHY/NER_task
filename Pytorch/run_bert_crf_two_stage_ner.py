import argparse
from datasetReader.DatasetReader import TwostageDataReader
from models.bert_for_ner import BertCrfTwoStageForNer
from transformers import BertConfig
from torch.utils.data import DataLoader,RandomSampler
from torch import optim
from tools.finetune_argparse import get_argparse
from tqdm import tqdm


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
              {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
               'weight_decay': args.weight_decay, 'lr': args.learning_rate},
              {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
               'lr': args.learning_rate},

              {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
               'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
              {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
               'lr': args.crf_learning_rate},

              {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)],
               'weight_decay': args.weight_decay, 'lr': args.crf_learning_rate},
              {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
               'lr': args.crf_learning_rate}
       ]

       optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate,eps=args.adam_epsilon )
       early_stop_step = 20000
       last_improve = 0  # 记录上次提升的step
       flag = False  # 记录是否很久没有效果提升
       dev_best_acc = 0
       correct = 0
       total = 0
       global_step = 0


       model.to(args.device)
       model.train()
       for epoch in tqdm(range(args.epochs),desc='Epoch'):
              for step,batch in enumerate(tqdm(train_dataloader,desc='Iteraction')):
                     optimizer.zero_grad()
                     batch = tuple(t.to(args.device) for t in batch)
                     inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'bieso_labels': batch[2],
                               'att_labels': batch[3]}
                     outputs = model(**inputs)


                     loss = outputs[0]
                     loss.backward()
                     optimizer.step()
                     global_step += 1

                     biesos_logits,att_logits = outputs[1],outputs[2]

                     biesos_tag,att_tag = model.crf_bieso.decode(biesos_logits,inputs['attention_mask']),model.crf_att.decode(att_logits,inputs['attention_mask'])







def main():



       args = get_argparse().parse_args()
       print(args)

       vocab_att_path = args.data_dir + '/vocab_attr_list.txt'

       with open(vocab_att_path,'r') as f:
              atts = f.readlines()
       vocab_len = len(atts)
       print('vocab_len',vocab_len)
       config = BertConfig.from_pretrained(args.bert_model_path)
       config.bieso_num_labels = 5
       config.att_num_labels = vocab_len

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



