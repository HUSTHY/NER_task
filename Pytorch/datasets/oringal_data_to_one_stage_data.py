import json
import os
import matplotlib.pyplot as plt
import time


def get_text_bios_atts(file_path,save_path):
    texts = []
    bios_list = []
    # attris_list = []
    vocab_att = []
    vocab_bios = []
    with open(file_path,'r') as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        text = line.get('text','')
        label = line.get('label','')
        bios,vocabulary = get_bios(text,label)
        # attris, att_list = get_attris(text, label)
        texts.append(text)
        bios_list.append(bios)
        vocab_bios.extend(vocabulary)
        # vocab_att.extend(att_list)

    text_save_path = save_path + '_text.txt'
    bios_save_path = save_path + '_bios.txt'
    # attris_save_path = save_path + '_attris.txt'

    with open(text_save_path,'w') as f:
        for line in texts:
            f.write(line+'\n')

    with open(bios_save_path,'w') as f:
        for line in bios_list:
            line = ' '.join(line)
            f.write(line+'\n')

    # with open(attris_save_path,'w') as f:
    #     for line in attris_list:
    #         line = ' '.join(line)
    #         f.write(line+'\n')

    return vocab_bios

def get_texts(file_path, save_path):
    texts = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        text = line.get('text', '')
        texts.append(text)
    text_save_path = save_path + '_text.txt'
    with open(text_save_path, 'w') as f:
        for line in texts:
            f.write(line + '\n')


# def get_atts(text,label):
#     att_list = []
#     attris = ['NULL'] * len(text)
#     for key,entitiys in label.items():
#         att_list.append(key)
#         index_start_and_ends = []
#         for k,v in entitiys.items():
#             index_start_and_ends.extend(v)
#         for start_end in index_start_and_ends:
#             start = int(start_end[0])
#             end = int(start_end[1])
#             for i in range(start,end+1):
#                 attris[i] = key
#     return attris,att_list

# def get_bios(text,label):
#     """ 根据文本和标签，把对应的每个字的BIEOS标签找到 """
#     bios = ['O']* len(text)
#     index_start_and_ends = []
#     for _,entitiys in label.items():
#         for _,v in entitiys.items():
#             index_start_and_ends.extend(v)
#     for start_end in index_start_and_ends:
#         start = int(start_end[0])
#         end = int(start_end[1])
#         if end-start == 0:
#             bios[start] = 'S'
#         if end-start == 1:
#             bios[start] = 'B'
#             bios[end] = 'E'
#         if end-start >= 2:
#             bios[start] = 'B'
#             for i in range(start+1,end):
#                 bios[i] = 'I'
#             bios[end] = 'E'
#     return bios

def get_bios(text,label):
    """ 根据文本和标签，把对应的每个字的BIOS标签找到 """
    bios = ['O']* len(text)
    vocabulary = []
    for name,entitiys in label.items():
        index_start_and_ends = []
        for _,v in entitiys.items():
            index_start_and_ends.extend(v)
        for start_end in index_start_and_ends:
            start = int(start_end[0])
            end = int(start_end[1])
            if end-start == 0:
                bios[start] = 'S_'+ name
                vocabulary.append(bios[start])
            if end-start >= 1:
                bios[start] = 'B_'+ name
                vocabulary.append(bios[start])
                for i in range(start + 1, end+1):
                    bios[i] = 'I_' + name
                    vocabulary.append(bios[i])
    vocabulary = list(set(vocabulary))
    return bios,vocabulary



def get_attris(text,label):
    att_list = []
    attris = ['NULL'] * len(text)
    for key,entitiys in label.items():
        att_list.append(key)
        index_start_and_ends = []
        for k,v in entitiys.items():
            index_start_and_ends.extend(v)
        for start_end in index_start_and_ends:
            start = int(start_end[0])
            end = int(start_end[1])
            for i in range(start,end+1):
                attris[i] = key
    return attris,att_list

def statistics_text_length(dir_path):
    dir_path = 'oringal_data'
    files = os.listdir(dir_path)
    text_lengths = []
    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path,'r') as f:
            lines = f.readlines()
        for line in lines:
            text = json.loads(line).get('text','')
            text_lengths.append(len(text))
    plt.hist(text_lengths,bins=50,color='blue')
    plt.show()

    text_lengths.sort(reverse=True)
    print('the max length of text is ',text_lengths[0])



def main():
    dir_path = 'oringal_data'
    files = os.listdir(dir_path)
    vocab_bios_list = []
    # vocab_att_list = []
    for file in files:
        if 'test' not in file:
            file_path = os.path.join(dir_path, file)
            save_path = os.path.join('one_stage_data',file.split('.')[0])
            print('save_path',save_path)
            vocab_bios = get_text_bios_atts(file_path,save_path)
            vocab_bios_list.extend(vocab_bios)
        else:
            file_path = os.path.join(dir_path, file)
            save_path = os.path.join('one_stage_data', file.split('.')[0])
            print('save_path', save_path)
            get_texts(file_path, save_path)
    vocab_bios_list = list(set(vocab_bios_list))
    vocab_bios_list.sort()

    # vocab_att_list = list(set(vocab_att_list))

    # with open('one_stage_data/vocab_bios_list.txt','w') as f:
    #     f.write('O'+'\n')
    #     for bios in vocab_bios_list:
    #         f.write(bios+'\n')

    # with open('two_stage_data/vocab_attr_list.txt','w') as f:
    #     f.write('NULL'+'\n')
    #     for attri in vocab_att_list:
    #         f.write(attri+'\n')

    statistics_text_length(dir_path)

if __name__ == '__main__':
    main()







