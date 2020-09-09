import json
import os


def get_text_bio_attri(file_path,save_path):
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            text = line.get('text','')
            label = line.get('label','')
            bios = get_bios(text,label)
            attris = get_attris(text,label)


def get_bios(text,label):
    """ 根据文本和标签，把对应的每个字的BIO标签找到 """
    bios = ['O']* len(text)
    index_start_and_ends = []
    for _,entitiys in label.items():
        for _,v in entitiys.items():
            index_start_and_ends.extend(v)
    for start_end in index_start_and_ends:
        start = int(start_end[0])
        end = int(start_end[1])
        bios[start] = 'B'
        for i in range(start+1,end+1):
            bios[i] = 'I'
    return bios


def get_attris(text,label):
    attris = ['NULL'] * len(text)
    index_start_and_ends = []
    for _,entitiys in label.items():
        for _,v in entitiys.items():
            index_start_and_ends.extend(v)
    for start_end in index_start_and_ends:
        start = int(start_end[0])
        end = int(start_end[1])
    return attris





def main():
    dir_path = 'oringal_data'
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        save_path = os.path.join('two_stage_data',file.split('.')[0])
        get_text_bio_attri(file_path,save_path)

if __name__ == '__main__':
    main()







