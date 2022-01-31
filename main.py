from async_timeout import enum
from sklearn.utils import shuffle
import torch
import  torch.functional as F
from  model.bilstm_crf import  BERT_BiLSTM_CRF as Model
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
BATCH_SIZE=

def setseed(seed=0):
    torch.manual_seed(seed)  # 设置随机数种子，方便重现
    torch.cuda.manual_seed(seed)

def get_data_path(dataset,type=TRAIN_SET):  # 

    datapath=os.path.join(DATA_DIR,dataset,DATASET_INFO[dataset][type])
    return os.path.abspath(datapath)


def train(dataset,type=TRAIN_SET,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    tokenizer=BertTokenizer.from_pretrained(BERT_NAME,do_lower_case=True)
    bert_config = BertConfig.from_pretrained(BERT_NAME)
    datapath=get_data_path(dataset)
    train_data=NERDataset(datapath,tokenizer,MAX_SEQUECE_LEN)
    dataloader=DataLoader(train_data,batch_size=BATCH_SIZE,sampler=RandomSampler(train_data))
    id2tag,tag2id=train_data.get_tags_ids()
    model=Model(bert_config)
    model.to(device)
   
    for i, (token_ids,token_type_ids,attention_mask,tagids,ori_text) in enumerate(dataloader):
        print(token_ids)

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
    train(Opt.dataset,device)



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