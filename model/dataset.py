import torch
from  torch.utils.data import Dataset,DataLoader
from tqdm import tqdm 

'''
数据是按句处理的，句子中每个汉字或符号都有一个tag, 表明实体类型等。 句之间用空行隔开
中 B-GE
国 I-GE 
。 O

闰  B-PER
土  I_PER
。  O
'''

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, token_type_ids, attention_mask, label_id, ori_tokens=""):
        """
        :param input_ids:       单词在词典中的编码
        :param attention_mask:  指定 对哪些词 进行self-Attention操作
        :param token_type_ids:  区分两个句子的编码（上句全为0，下句全为1）
        :param label_id:        标签的id
        """
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_id = label_id
        self.ori_tokens = ori_tokens

class NERDataset(Dataset):
    def __init__(self,filename,tokenizer,max_seq_len):
        super(NERDataset,self).__init__()
        self.data=[]
 
        self.alltags=set()
         
        self.ori_texts=[]
        self.featues=[]
        self.tag2id={}

        self.tokenizer=tokenizer
        self.max_seq_len=max_seq_len
        self.id2tag=[]
        self.readfile(filename)
      

    
        
    
    def __getitem__(self, index):
        return self.featues[index].input_ids,self.featues[index].token_type_ids,self.featues[index].attention_mask,self.featues[index].label_id,self.featues[index].ori_tokens
    
    def __len__(self):
        return len(self.featues)

    def readfile(self,filename):
        ori_texts=[]
        with open(filename,encoding="utf-8") as f:
            thisline = []
            thistag = []
            for line in tqdm(f,desc="reading samples"):
           
                line=line.strip().split()
                if(len(line)>0):
                   # tokenid= self.tokenizer.tokenize(line[0])
                    thisline.append(line[0])
                    thistag.append(line[1])
                    self.alltags.add(line[1])
                else:
                    # process every single line 
                    # preprocess thisline
                    if len(thisline) > (self.max_seq_len-2):   # 对短的截短即可，下面会自动填充
                        thisline=thisline[:self.max_seq_len-2]
                    sen_code=self.tokenizer.encode_plus(thisline,add_special_tokens=True,max_length=self.max_seq_len,padding="max_length")
                    input_ids, token_type_ids, attention_mask = sen_code["input_ids"], sen_code["token_type_ids"], sen_code["attention_mask"]
                    ## preprocess tags
                    if len(thistag) >(self.max_seq_len-2):
                        thistag=thistag[:self.max_seq_len-2]
                        thistag=["O"]+thistag+["O"]
                    else:
                        thistag.extend(["O"]* (self.max_seq_len-2-len(thistag)))
                        thistag=["O"]+thistag+["O"]

                    ori_text=self.tokenizer.decode(input_ids)
                    ori_texts.append(ori_text)
                    self.featues.append(InputFeatures(input_ids,token_type_ids,attention_mask,thistag,ori_text))
                    
                    thisline=[]
                    thistag=[]
        
        #
        self.tag2id={tag:i for i,tag in enumerate(self.alltags) }
        self.id2tag= list(self.alltags)
        # convert  labels to Ids.
        self.ori_texts=ori_texts
        _,tag2id = self.get_tags_ids()
        for index,feature in enumerate(tqdm(self.featues)):
            for id in range(len(feature.label_id)):
                self.featues[index].label_id[id] = tag2id[self.featues[index].label_id[id]]
     
        print("...")    
    def get_tag_num(self):
        return len(self.tags)

    def get_tags_ids(self):
        
        return self.id2tag,self.tag2id
         
    def get_label(self):
        return self.alltags

    def get_ori_tokens(self):
        return self.ori_texts