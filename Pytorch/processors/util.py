from tqdm import tqdm
import time

def file_read(path):
    with open(path,'r') as f:
        lines = f.readlines()
    for line in lines:
        yield line.strip('\n')
    # lines = [line.strip('\n') for line in lines]
    # return lines

# def bieso_label_to_id(biesos,max_sentence_length):
#     label_map = { label:id for id,label in enumerate(['O','B','I','E','S'])} #'O'放在第一位是为了后面labels padding好操作
#     biesos_labelid = []
#     for bieso in tqdm(biesos,desc='biesos_label_to_id'):
#         label = []
#         for ele in bieso.split(' '):
#             label.append(label_map[ele])
#         label_padding = [0]*(max_sentence_length-len(label)) #做padding操作
#         label.extend(label_padding)
#         biesos_labelid.append(label)
#     return biesos_labelid

def bieso_label_to_id(biesos,max_sentence_length):
    label_map = { label:id for id,label in enumerate(['X','B','I','S','O'])} #'O'放在最后一位不能放在第一位
    biesos_labelid = []
    all_input_lens = []
    for bieso in tqdm(biesos,desc='biesos_label_to_id'):
        label = []
        for ele in bieso.split(' '):
            label.append(label_map[ele])
        all_input_lens.append(len(label))
        label_padding = [0]*(max_sentence_length-len(label)) #做padding操作
        label.extend(label_padding)
        biesos_labelid.append(label)
    return biesos_labelid,all_input_lens


def atts_label_to_id(atts,vocab_att_path,max_sentence_length):
    with open(vocab_att_path,'r') as f:
        vocab_atts = [ line.strip('\n') for line in f.readlines()]
    label_map = {label: id for id, label in enumerate(vocab_atts)}


    atts_labelid = []
    for att in tqdm(atts,desc='atts_label_to_id'):
        label = []
        for ele in att.split(' '):
            label.append(label_map[ele])
        label_padding = [0] * (max_sentence_length - len(label)) #做padding操作
        label.extend(label_padding)
        atts_labelid.append(label)
    return atts_labelid


def bios_label_to_id_one_stage(bios,vocab_bios_path,max_sentence_length):
    with open(vocab_bios_path, 'r') as f:
        vocab_atts = [line.strip('\n') for line in f.readlines()]
    label_map = {label: id for id, label in enumerate(vocab_atts)}

    bios_labelid = []
    all_input_lens = []
    for k in tqdm(bios, desc='biesos_label_to_id'):
        label = []
        label.append(label_map['O'])
        for ele in k.split(' '):
            label.append(label_map[ele])
        label.append(label_map['O'])
        all_input_lens.append(len(label))
        if len(label)> max_sentence_length:
            label = label[0:max_sentence_length]
        label_padding = [0] * (max_sentence_length - len(label))  # 做padding操作
        label.extend(label_padding)
        bios_labelid.append(label)
    return bios_labelid,all_input_lens
