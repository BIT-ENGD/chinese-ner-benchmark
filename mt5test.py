from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch 
import torch.nn.functional as F
import torch.nn as nn 

MODELNAME="google/mt5-small" #"google/mt5-xxl" # "google/mt5-large"  #"google/mt5-base" "google/mt5-xxl" "google/mt5-xl"
model = MT5ForConditionalGeneration.from_pretrained(MODELNAME)
tokenizer = T5Tokenizer.from_pretrained(MODELNAME)

class SoftEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                n_tokens: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        """这个类用来给模型附加一个用于学习的embedding
        Args:
            wte (nn.Embedding): 这个参数，是预训练模型的embedding，载入进来用来提取一些参数。
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens, 
                                                                                  random_range, 
                                                                                  initialize_from_vocab))

    def initialize_embedding(self, 
                             wte: nn.Embedding,
                             n_tokens: int = 10, 
                             random_range: float = 0.5, 
                             initialize_from_vocab: bool = True):
        """初始化学习向量
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        # 有两种初始化方式，一种是从预训练模型copy一部分token，进行训练
        # 另一种是随机生成一部分训练
        # 结果上来说区别不大
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        # 把我们新加入的固定长度的，用于代表任务的prompt embedding，和实际的embedding合并
        return torch.cat([learned_embedding, input_embedding], 1)


n_tokens = 100
s_wte = SoftEmbedding(model.get_input_embeddings(), 
                      n_tokens=n_tokens, 
                      initialize_from_vocab=True)
# 用我们设计的类，替换原来的embedding层
model.set_input_embeddings(s_wte)

if torch.cuda.is_available():
    model = model.cuda()

# 把除了第0个，就是我们要训练的prompt embedding以外的参数，都设置为不需要梯度
parameters = list(model.parameters())
for x in parameters[1:]:
    x.requires_grad = False