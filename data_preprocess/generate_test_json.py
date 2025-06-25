import json
import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

labels_map = {
    'MIntRec': {
        'intent': ['complain', 'praise', 'apologise', 'thank', 'criticize', 'agree', 'taunt', 'flaunt', 'joke',
                   'oppose', 'comfort', 'care', 'inform', 'advise', 'arrange', 'introduce', 'leave', 'prevent',
                   'greet', 'ask for help']
    },
    'MIntRec2.0': {
        'intent': ['acknowledge', 'advise', 'agree', 'apologise', 'arrange',
                   'ask for help', 'asking for opinions', 'care', 'comfort', 'complain',
                   'confirm', 'criticize', 'doubt', 'emphasize', 'explain',
                   'flaunt', 'greet', 'inform', 'introduce', 'invite',
                   'joke', 'leave', 'oppose', 'plan', 'praise',
                   'prevent', 'refuse', 'taunt', 'thank', 'warn']
    },
    "MELD": {
        "emotion": ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'anger', 'disgust'],
        "sentiment": ['neutral', 'positive', 'negative']
    },
    "MELD-DA": {
        "dialogue_act": ['greeting', 'question', 'answer', 'statement-opinion', 'statement-non-opinion', 'apology',
                         'command', 'agreement', 'disagreement', 'acknowledge', 'backchannel', 'others']
    },
    "IEMOCAP": {
        'emotion': ['angry', 'happy', 'sad', 'neutral', 'frustrated', 'excited']
    },
    "IEMOCAP-DA": {
        'dialogue_act': ['greeting', 'question', 'answer', 'statement-opinion', 'statement-non-opinion', 'apology',
                         'command', 'agreement', 'disagreement', 'acknowledge', 'backchannel', 'others']
    },
    'Ch-sims': {
        'sentiment_regression': ['-1.0', '-0.8', '-0.6', '-0.4', '-0.2', '0.0', '0.2', '0.4', '0.6', '0.8', '1.0'],
        'sentiment': ['neutral', 'positive', 'negative'],
    },
    'UR-FUNNY': {
        'speaking_style': ['humorous', 'serious']
    },
    'MUStARD': {
        'speaking_style': ['sincere', 'sarcastic']
    },
    'MOSI': {
        # 'sentiment': ['neutral', 'positive', 'negative'],
        'sentiment': ['positive', 'negative'], # neutral 归为positive

    },
    "AnnoMi-therapist":{
        "communication_behavior":['question', 'therapist_input', 'reflection', 'other']
    },
    "AnnoMi-client":{
        "communication_behavior":['neutral', 'change', 'sustain']
    },
}

task_map = {
    'MIntRec': ['intent'],
    'MIntRec2.0': ['intent'],
    'MELD': ['emotion'],
    'MELD-DA': ['dialogue_act'],
    'Ch-sims': ['sentiment'],
    'MOSI': ['sentiment'],
    'IEMOCAP': ['emotion'],
    'IEMOCAP-DA':['dialogue_act'],
    'UR-FUNNY':["speaking_style"],
    "AnnoMi-therapist":["communication_behavior"],
    "AnnoMi-client":["communication_behavior"],
    'MUStARD':['speaking_style'],
}

def generate_instruction_data(dataset_name, data_path, file, task):
    # 读取TSV文件
    tsv_file = os.path.join(data_path, file)  # 替换为你的TSV文件路径
    print(tsv_file)
    df = pd.read_csv(tsv_file, sep='\t')
    video_files = os.listdir(os.path.join(data_path, 'video'))

    # 初始化JSON结构
    json_data = []
    cnt = 0
    # 生成JSON数据
    for index, row in df.iterrows():
        idx = row['id']
        utterance = row['utterance']
        label = row[task]

        video_file = f"{dataset_name}_{idx}" + '.mp4'
        # print('video_file: {}'.format(video_file))
            
        conversation_parts = []
        conversation_parts.append({
            "from": "human",
            "value": f"{utterance}"
        })

        conversation_parts.append({
            "from": "gpt",
            "value": str(label).lower()
        })

        json_data.append({
            "id": index,
            "video": video_file,
            "conversations": conversation_parts
        })
    
    return json_data

if __name__ == '__main__':

    root_dir = '/root/zhanghanlei/'
    work_dir = 'workspace/test'
    data_dir = 'Datasets'

    datasets = ['MIntRec', 'MIntRec2.0','MELD','MELD-DA',"IEMOCAP",'IEMOCAP-DA','Ch-sims', 'MUStARD','MOSI','UR-FUNNY','AnnoMi-therapist','AnnoMi-client']

    for dataset in datasets:
        
        data_path = os.path.join(root_dir, data_dir, dataset)
        tasks = task_map[dataset]

        for task in tasks:
            labels_dict = labels_map[dataset]

            json_data = generate_instruction_data(dataset, data_path, 'test.tsv', task)

            save_name = 'test_' + task + '.json'
            save_path = os.path.join(root_dir, work_dir, 'data', save_name)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            
            with open(save_path, "w", encoding='utf-8') as outfile:
                json.dump(json_data, outfile, ensure_ascii=False, indent=4)






        
