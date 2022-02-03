from async_timeout import enum
#from black import out
#from sklearn.utils import shuffle
import torch
import  torch.functional as F
from  model.bilstm_crf import  BERT_BiLSTM_CRF_Ex as Model
from model.dataset import NERDataset
import optparse
import pickle
import logging 
from  tensorboardX  import SummaryWriter
from tqdm  import tqdm, trange
import os 
from  torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
from transformers import  AdamW, get_linear_schedule_with_warmup,BertTokenizer,BertConfig
#import conlleval as evaluate

# https://blog.csdn.net/weixin_42001089/article/details/97657149  bert实体关系解读

## 英文实现参考： https://www.freesion.com/article/3725642002/


# NER  演示： https://www.nuomiphp.com/github/zh/5ff465d996e3df1f3325cd35.html

# bert bilstm-crf NER  https://gitee.com/shimii/BERT-BiLSTM-CRF-NER-pytorch  (中文)


# 中文实体识别： https://blog.csdn.net/baqnliaozhihui/article/details/109244094


## 中文ner  https://github.com/Bureaux-Tao/BiLSTM-NER-PyTorch


## 简化代码： 参考 https://github.com/positivepeng/nlp-beginner-projects/tree/master/project3-Named%20Entity%20Recognition

###  https://zhuanlan.zhihu.com/p/346828049   一个可以作为base line的库


'''
logging.debug(u"debug")
logging.info(u"info")
logging.warning(u"warning")
logging.error(u"error")
logging.critical(u"critical")
'''

DATASET_INFO={"MSRA":["train_dev.char.bmes","test.char.bmes","test.char.bmes"],"ResumeNER":["train.char.bmes","dev.char.bmes","test.char.bmes"],"WeiboNER":["train.all.bmes","dev.all.bmes","test.all.bmes"]}
DATA_DIR="data"


# hyper paramters
TRAIN_SET=0
VALID_SET=1
TEST_SET=2
MAX_SEQUECE_LEN=128   #最大长度
BERT_NAME="bert-base-chinese"
BATCH_SIZE=128 #64

DATASET_TRAIN_FILE="dataset.dat"
DATASET_VALID_FILE="dataset_valid.dat"
DATASET_TEST_FILE="dataset_test.dat"
RNN_DIM=128
NEED_BIRNN=True
LR_RATE=3e-05
ADM_EPSILON=1e-08

WARMUP_STEPS=0
GRADIENT_ACCUMULATION_STEPS=1
EPOCH_NUM=10
LOGGIN_STEPS=500
OUTPUT_DIR="output"
DO_TRAIN=True
DO_EVAL=True
DO_TEST=True
def setseed(seed=0):
    torch.manual_seed(seed)  # 设置随机数种子，方便重现
    torch.cuda.manual_seed(seed)

def get_data_path(dataset,type=TRAIN_SET):  # 

    filename=DATASET_INFO[dataset][type]
    if(filename == ""):
       filename=DATASET_INFO[dataset][TEST_SET] # for which it has no specified value.
    datapath=os.path.join(DATA_DIR,dataset,filename)
    return os.path.abspath(datapath)

def  save_var(var,filename):
    with open(filename,"wb") as f:
        pickle.dump(var,f)

def  load_var(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)


def collate(data):  #转换变量类型
    
    input_ids, token_type_ids, attention_mask,label_id,ori_tokens = list(zip(*data))
    new_input_ids = torch.LongTensor(input_ids)
    new_token_type_ids = torch.LongTensor(token_type_ids)
    new_attention_mask = torch.LongTensor(attention_mask)
    new_label_id = torch.LongTensor(label_id)
    return (new_input_ids, new_token_type_ids,new_attention_mask,new_label_id),ori_tokens

def do_valid(dataset,model,tokenizer,device,writer):
    ori_labels, pred_labels = [], []
    model.eval()
    datapath=get_data_path(dataset,VALID_SET)

    if os.path.exists(DATASET_VALID_FILE):
       valid_data= load_var(DATASET_VALID_FILE)
    else:
        valid_data=NERDataset(datapath,tokenizer,MAX_SEQUECE_LEN)
        save_var(valid_data,DATASET_VALID_FILE)
    id2tag,tag2id=valid_data.get_tags_ids()
    dataloader=DataLoader(valid_data,batch_size=BATCH_SIZE,collate_fn=collate,sampler=SequentialSampler(valid_data))
    new_ori_text=[]
    for step, batch in enumerate(tqdm(dataloader,desc="Valid DataLoader Progress")):
        batch,ori_text=batch
        batch = tuple(t.to(device) for t in batch)
        token_ids,token_type_ids,attention_mask,tagids=batch
        with torch.no_grad():
            logits=model.predict(token_ids,token_type_ids,attention_mask)
        for l in logits:
            pred_labels.append([id2tag[idx] for idx in l])
        for l in tagids:
            ori_labels.append([id2tag[idx.item()] for idx in l])
        new_ori_text.extend(ori_text)
    eval_list = []
    all_ori_tokens=valid_data.get_ori_tokens()
    for ori_tokens, oril, prel in zip(all_ori_tokens, ori_labels, pred_labels):
        for ot, ol, pl in zip(ori_tokens, oril, prel):
            if ot in ["[CLS]", "[SEP]"]:
                continue
            eval_list.append(f"{ot} {ol} {pl}\n")
        eval_list.append("\n")

def do_test(dataset,model,tokenizer,device,writer):
    ori_labels, pred_labels = [], []
    model.eval()
    datapath=get_data_path(dataset,TEST_SET)

def do_train(dataset,type=TRAIN_SET,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    writer = SummaryWriter(logdir=os.path.join(OUTPUT_DIR, "eval"), comment="chinese-ner")
    datapath=get_data_path(dataset)
    tokenizer=BertTokenizer.from_pretrained(BERT_NAME,do_lower_case=True)
    if os.path.exists(DATASET_TRAIN_FILE):
       train_data= load_var(DATASET_TRAIN_FILE)
    else:
        train_data=NERDataset(datapath,tokenizer,MAX_SEQUECE_LEN)
        save_var(train_data,DATASET_TRAIN_FILE)
    num_labels=len(train_data.get_label())
    bert_config = BertConfig.from_pretrained(BERT_NAME,num_labels=num_labels)
    model=Model.from_pretrained(BERT_NAME,config=bert_config,rnn_dim=RNN_DIM,need_birnn=NEED_BIRNN)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR_RATE, eps=ADM_EPSILON)
    dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,collate_fn=collate,sampler=RandomSampler(train_data))
    t_total = len(dataloader) // GRADIENT_ACCUMULATION_STEPS * EPOCH_NUM
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)

    id2tag,tag2id=train_data.get_tags_ids()
    
    model.to(device)
    
    global_step, tr_loss, logging_loss, best_f1 = 0, 0.0, 0.0, 0.0
    for epoch in tqdm(range(EPOCH_NUM),desc="Epoch Progress"):
        model.train()
        for step, batch in enumerate(tqdm(dataloader,desc="Train DataLoader Progress")):
            batch,ori_text=batch
            batch = tuple(t.to(device) for t in batch)
            token_ids,token_type_ids,attention_mask,tagids=batch
            outputs=model(token_ids,tagids,token_type_ids,attention_mask)
            loss=outputs
            if GRADIENT_ACCUMULATION_STEPS > 1:
                        loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            tr_loss+=loss.item()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # 更新参数
                optimizer.step()
                scheduler.step()
                # 梯度清零
                model.zero_grad()
                global_step += 1
                if LOGGIN_STEPS > 0 and global_step % LOGGIN_STEPS == 0:
                    tr_loss_avg = (tr_loss - logging_loss) / LOGGIN_STEPS
                    writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                    logging_loss = tr_loss
        if DO_EVAL:
           do_valid(dataset,model,tokenizer,device,writer)

    if DO_TEST:
        do_test(dataset,model,tokenizer,device,writer)



  

def main(Opt):
    print(Opt.model)
    if( Opt.device == "cuda"):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
    #lstmcrfModel=Model()

    if DO_TRAIN:
        do_train(dataset=Opt.dataset,device=device)



if __name__ == "__main__":
   # logging.basicConfig(level=logging.NOTSET) ## 打印所有日志

    
    optparser = optparse.OptionParser()
    optparser.add_option(
    "-d", "--dataset", default="MSRA",
    help="set the name of dataset"
     )
    optparser.add_option(
    "-m","--model",default="bilstmcrf"
    )

    optparser.add_option(
    "-s","--device",default="cuda"
    )

    opts = optparser.parse_args()[0]

    # set random_seed
    setseed()
    main(opts)