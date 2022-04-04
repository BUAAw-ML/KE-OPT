import torch
import os
import numpy as np



### v2.0-singlevd contains a video encoder(VIT-base-32) and a multimodal encoder(bert-base-uncased)


###process videoswin-32 weight 

def trans(x):
    return torch.from_numpy(x)
opt_weight={}
swin_weight = torch.load("./output/pretrianed_weights/ckpt_video-swin.pt")

for k,v in swin_weight.items():
    opt_weight['opt.video_encoder.' + k] = v



bert_weight = torch.load("./output/pretrianed_weights/bert-base-uncased.bin")

### word_embedding_weights:
opt_weight['opt.txt_embeddings.word_embeddings.weight'] = bert_weight['bert.embeddings.word_embeddings.weight']
### position_embedding weights:
opt_weight['opt.txt_embeddings.position_embeddings.weight'] = bert_weight['bert.embeddings.position_embeddings.weight']

opt_weight['opt.txt_embeddings.layernorm.weight'] = bert_weight['bert.embeddings.LayerNorm.gamma']
opt_weight['opt.txt_embeddings.layernorm.bias']  = bert_weight['bert.embeddings.LayerNorm.beta']

for  i in range(12):
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.0.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.query.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.0.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.query.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.1.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.key.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.1.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.key.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.2.weight'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.value.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.2.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.self.value.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.3.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.dense.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.attention.linears.3.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.dense.bias'] 
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.ff_layer.linear1.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.intermediate.dense.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.ff_layer.linear1.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.intermediate.dense.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.ff_layer.linear2.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.dense.weight']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.ff_layer.linear2.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.dense.bias']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.layernorm1.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.gamma']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.layernorm1.bias']  = bert_weight['bert.encoder.layer.'+str(i)+'.attention.output.LayerNorm.beta']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.layernorm2.weight']  = bert_weight['bert.encoder.layer.'+str(i)+'.output.LayerNorm.gamma']
    opt_weight['opt.txt_encoder.layer.'+str(i)+'.layernorm2.bias'] = bert_weight['bert.encoder.layer.'+str(i)+'.output.LayerNorm.beta']
    



opt_weight['cls.dense.weight']  = bert_weight['cls.predictions.transform.dense.weight']
opt_weight['cls.dense.bias']  = bert_weight['cls.predictions.transform.dense.bias']
opt_weight['cls.layernorm.weight'] = bert_weight['cls.predictions.transform.LayerNorm.gamma' ]
opt_weight['cls.layernorm.bias'] =bert_weight['cls.predictions.transform.LayerNorm.beta']
opt_weight['cls.decoder.weight'] = bert_weight['cls.predictions.decoder.weight']
opt_weight['cls.decoder.bias'] = bert_weight['cls.predictions.bias']




if not os.path.exists('./output/pretrianed_weights'):
    os.makedirs('./output/pretrianed_weights')
torch.save(opt_weight,'./output/pretrianed_weights/double_modify_videoswinbase32_bertbaseuncased.pt')





