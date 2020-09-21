import argparse

def get_argparse_two_stage():
    parser = argparse.ArgumentParser(description='init params configuration')
    parser.add_argument('--max_sentence_length', type=int, default=50)
    parser.add_argument('--bert_model_path', type=str, default='pre_train_model/roberta')
    parser.add_argument('--data_dir', type=str, default='datasets/two_stage_data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--learning_rate", default=3e-5, type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=5e-5, type=float,help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--epochs", default=200, type=int, help="Train epochs iterations.")
    parser.add_argument('--device',default='cuda',type=str, help="gpu cuda")
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help="gpu cuda")
    return parser


def get_argparse_one_stage():
    parser = argparse.ArgumentParser(description='init params configuration')
    parser.add_argument('--max_sentence_length', type=int, default=50)
    parser.add_argument('--bert_model_path', type=str, default='pre_train_model/Chinese-BERT-wwm')
    parser.add_argument('--data_dir', type=str, default='datasets/one_stage_data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--learning_rate", default=2e-5, type=float,help="The initial learning rate for Adam.")
    parser.add_argument("--crf_learning_rate", default=2e-5, type=float,help="The initial learning rate for crf and linear layer.")
    parser.add_argument("--weight_decay", default=0.01, type=float,help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--epochs", default=200, type=int, help="Train epochs iterations.")
    parser.add_argument('--device',default='cuda',type=str, help="gpu cuda")
    parser.add_argument('--warmup_proportion', default=0.1, type=float, help="gpu cuda")
    return parser