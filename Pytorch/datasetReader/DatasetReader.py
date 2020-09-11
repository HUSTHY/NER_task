from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from tqdm import tqdm
import os
import logging
logger = logging.getLogger(__name__)
from processors.util import file_read,bios_label_to_id,atts_label_to_id

class TwostageDataReader(Dataset):
    def __init__(self,args,text_file_name,bios_file_name,atts_file_name,repeat = 1):
        self.max_sentence_length = args.max_sentence_length
        self.repeat = repeat
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        self.process_data_list = self.read_file(args.data_dir,text_file_name,bios_file_name,atts_file_name)

    def read_file(self,data_dir,text_file_name,bios_file_name,atts_file_name):
        process_data_list = []
        text_file_path = os.path.join(data_dir,text_file_name)
        bios_file_path = os.path.join(data_dir,bios_file_name)
        atts_file_path = os.path.join(data_dir,atts_file_name)

        texts = file_read(text_file_path)
        bio_labels = []
        att_labels = []
        if os.path.exists(bios_file_path) and os.path.exists(atts_file_path):
            bios = file_read(bios_file_path)
            atts = file_read(atts_file_path)

            bios_labelid = bios_label_to_id(bios)
            vocab_att_path = data_dir+'/vocab_attr_list.txt'
            atts_labelid = atts_label_to_id(atts,vocab_att_path)

            for label in tqdm(bios_labelid,desc='bio_labels to tensor'):
                label = torch.tensor(label,dtype=torch.long)
                bio_labels.append(label)

            for label in tqdm(atts_labelid, desc= 'att_labels to tensor'):
                label = torch.tensor(label, dtype=torch.long)
                att_labels.append(label)

        input_ids_list = []
        attention_mask_list = []
        for text in tqdm(texts,desc='text convert_into_indextokens_and_segment_id to tensor'):
            input_ids, attention_mask = self.convert_into_indextokens_and_segment_id(text)

            input_ids = torch.tensor(input_ids,dtype=torch.long)
            attention_mask = torch.tensor(attention_mask,dtype=torch.long)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)

        if len(bio_labels)>0 and len(att_labels)>0:
            for input_ids,attention_mask,bio_label,att_label in zip(input_ids_list,attention_mask_list,bio_labels,att_labels):
                process_data_list.append((input_ids,attention_mask,bio_label,att_label))
        else:
            for input_ids,attention_mask in zip(input_ids_list,attention_mask_list):
                process_data_list.append((input_ids,attention_mask,None,None))
        return process_data_list



    def convert_into_indextokens_and_segment_id(self,text):
        tokeniz_text = self.tokenizer.tokenize(text[0:self.max_sentence_length])
        input_ids = self.tokenizer.convert_tokens_to_ids(tokeniz_text)
        attention_mask = [1] * len(input_ids)

        pad_indextokens = [0] * (self.max_sentence_length - len(input_ids))
        input_ids.extend(pad_indextokens)
        attention_mask_pad = [0] * (self.max_sentence_length - len(attention_mask))
        attention_mask.extend(attention_mask_pad)

        return input_ids, attention_mask


    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.process_data_list)
        return data_len

    def __getitem__(self, item):
        input_ids = self.process_data_list[item][0]
        attention_mask = self.process_data_list[item][1]
        bio_label = self.process_data_list[item][2]
        att_label = self.process_data_list[item][3]
        return input_ids,attention_mask,bio_label,att_label
