from datasetReader.DatasetReader import OneStageDataReaderBilstmCrf
from models.bilstm_for_ner import BiLstmCRFNer
from tools.finetune_argparse import get_argparse_bilstmcrf_ner
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
from torch import optim
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

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


def train(model, train_data, dev_data, args):
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size, collate_fn=collate_fn)
    paramters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(paramters,lr = args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)

    dev_best_acc = 0
    global_step = 0
    model.to(args.device)
    model.train()
    # seed_everything(args.seed)

    for epoch in tqdm(range(args.epochs), desc='Epoch'):



def main():
    args = get_argparse_bilstmcrf_ner().parse_args()
    labels_path = args.data_dir + '/vocab_bios_list.txt'
    with open(labels_path, 'r') as f:
        bios = f.readlines()
    biso_labels = [ele.strip('\n') for ele in bios]
    print('vocab_len', biso_labels)

    vocab_path = args.bert_model_path+'/vocab.txt'
    with open(vocab_path, 'r') as f:
        vocab = f.readlines()
    vocab = [ele.strip('\n') for ele in vocab]
    print('vocab_len', vocab)

    vocab_size = len(vocab)
    embedding_size = args.embedding_size
    hidden_size = args.hidden_size
    drop_p = args.drop_p

    args.bios_id2label = {id: label for id, label in enumerate(biso_labels)}
    print(args)

    model = BiLstmCRFNer(vocab_size = vocab_size,embedding_size = embedding_size,hidden_size = hidden_size,biso_labels = biso_labels,drop_p = drop_p)

    train_data = OneStageDataReaderBilstmCrf(args=args, text_file_name='train_text.txt', bieso_file_name='train_bios.txt' )
    dev_data = OneStageDataReaderBilstmCrf(args=args, text_file_name='dev_text.txt', bieso_file_name='dev_bios.txt' )

    train(model, train_data, dev_data, args)

if __name__ == '__main__':
    main()