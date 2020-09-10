import argparse
from datasetReader.DatasetReader import TwostageDataReader


if __name__ == '__main__':
       parser = argparse.ArgumentParser(description='init params configuration')
       parser.add_argument('--max_sentence_length',type=int,default=50)
       parser.add_argument('--bert_model_path',type = str ,default='pre_train_model/roberta')
       parser.add_argument('--data_dir', type = str, default='datasets/two_stage_data')
       args = parser.parse_args()
       print(args)

       train_data = TwostageDataReader(args=args,text_file_name='train_text.txt',bios_file_name='train_bios.txt',atts_file_name ='train_attris.txt')


