from async_timeout import enum
from black import out
from sklearn.utils import shuffle
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
from  torch.utils.data import Dataset,DataLoader,RandomSampler
from transformers import BertTokenizer,BertConfig


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

DATASET_INFO={"MSRA":["train_dev.char.bmes","","test.char.bmes"],"ResumeNER":["train.char.bmes","dev.char.bmes","test.char.bmes"],"WeiboNER":["train.all.bmes","dev.all.bmes","test.all.bmes"]}
DATA_DIR="data"


TRAIN_SET=0
VALID_SET=1
TEST_SET=2
MAX_SEQUECE_LEN=128   #最大长度
BERT_NAME="bert-base-chinese"
BATCH_SIZE=16

DATA_SET_FILE="dataset.dat"
RNN_DIM=128
NEED_BIRNN=True
def setseed(seed=0):
    torch.manual_seed(seed)  # 设置随机数种子，方便重现
    torch.cuda.manual_seed(seed)

def get_data_path(dataset,type=TRAIN_SET):  # 

    datapath=os.path.join(DATA_DIR,dataset,DATASET_INFO[dataset][type])
    return os.path.abspath(datapath)

def  save_var(var,filename):
    with open(filename,"wb") as f:
        pickle.dump(var,f)

def  load_var(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)



def collate(data):
   
    input_ids, token_type_ids, attention_mask,label_id,ori_tokens = list(zip(*data))
  
    new_input_ids = torch.LongTensor(input_ids)
    new_token_type_ids = torch.LongTensor(token_type_ids)
    new_attention_mask = torch.LongTensor(attention_mask)
    new_label_id = torch.LongTensor(label_id)
    #new_ori_tokens = torch.LongTensor(ori_tokens)
  
   # return (input_ids, token_type_ids,attention_mask,label_id,ori_tokens)
    return (new_input_ids, new_token_type_ids,new_attention_mask,new_label_id),ori_tokens


def train(dataset,type=TRAIN_SET,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    datapath=get_data_path(dataset)
    tokenizer=BertTokenizer.from_pretrained(BERT_NAME,do_lower_case=True)
    if os.path.exists(DATA_SET_FILE):
       train_data= load_var(DATA_SET_FILE)
    else:
        train_data=NERDataset(datapath,tokenizer,MAX_SEQUECE_LEN)
        save_var(train_data,DATA_SET_FILE)
    num_labels=len(train_data.get_label())
    bert_config = BertConfig.from_pretrained(BERT_NAME,num_labels=num_labels)
    model=Model.from_pretrained(BERT_NAME,config=bert_config,rnn_dim=RNN_DIM,need_birnn=NEED_BIRNN)


    dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,collate_fn=collate,sampler=RandomSampler(train_data))
    id2tag,tag2id=train_data.get_tags_ids()
 
    model.to(device)


    for i, batch in enumerate(dataloader):
        batch,ori_text=batch
        batch = tuple(t.to(device) for t in batch)
        token_ids,token_type_ids,attention_mask,tagids=batch
        outputs=model(token_ids,tagids,token_type_ids,attention_mask)
        loss=outputs
        

def valid():
    pass 

def test():
    pass 

def main(Opt):
    print(Opt.model)
    if( Opt.device == "cuda"):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device=torch.device("cpu")
    #lstmcrfModel=Model()

    train(dataset=Opt.dataset,device=device)



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