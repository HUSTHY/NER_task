from tqdm import tqdm


def file_read(path):
    with open(path,'r') as f:
        lines = f.readlines()
    for line in lines:
        yield line.strip('\n')

def bios_label_to_id(bios):
    label_map = { label:id for id,label in enumerate(['B','I','O'])}
    bios_labelid = []
    for bio in tqdm(bios,desc='bios_label_to_id'):
        label = []
        for ele in bio.split(' '):
            label.append(label_map[ele])
        bios_labelid.append(label)
    return bios_labelid


def atts_label_to_id(atts,vocab_att_path):
    with open(vocab_att_path,'r') as f:
        vocab_atts = [ line.strip('\n') for line in f.readlines()]
    label_map = {label: id for id, label in enumerate(vocab_atts)}

    atts_labelid = []
    for att in tqdm(atts,desc='atts_label_to_id'):
        label = []
        for ele in att.split(' '):
            label.append(label_map[ele])
        atts_labelid.append(label)
    return atts_labelid
