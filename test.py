import torch 



a=torch.randn(10,1)
print(a)
b=torch.nn.Dropout(0.1)(a)
print("*"*9)
print(b)


import torch as t
from torch import nn

device =t.device("cuda" if t.cuda.is_available() else "cpu")
# in_features由输入张量的形状决定，out_features则决定了输出张量的形状 
connected_layer = nn.Linear(in_features =64*64*3, out_features =1,device=device)

# 假定输入的图像形状为[64,64,3]
input = t.randn(3,64,64,3)

# 将四维张量转换为二维张量之后，才能作为全连接层的输入
input = input.view(3,-1).to(device)
print(input.shape)
output = connected_layer(input) # 调用全连接层
print(output.shape)
print(output.data[0][0])


s = ["1", "2", "3", "4", "5"]
t = ["a", "b", "c", "d", "e"]
 
for x, y in zip(s, t):
    print(x+"-"+y)
 
print("######")
 
u = {"1", "2", "3", "4", "5"}
v = {"a", "b", "c", "d", "e"}
 
for x in zip(u, v):
    print(x)


t=torch.eye(5,6)
print(t)

LR_RATE=3e-05  #0.00003

print(0.00003*100000)