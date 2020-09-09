import json
import os

def get_text_bio_attri(file_path):
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            text = line.get('text','')
            label = line.get('label','')


def main():
    dir_path = 'datasets/oringal_data'
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        get_text_bio_attri(file_path)

if __name__ == '__main__':
    main()







