import transformers
from transformers import XLNetTokenizer, XLNetModel
from transformers import AutoModel, BertTokenizerFast


import torch
import torch.nn as nn
import torch.nn.functional as F






class LateFusion(nn.Module):
    def __init__(self, in1=768, in2=200, hidden_size=500, out = 768):
        super(LateFusion, self).__init__()
        
        self.embedding1 = nn.Linear(in1, hidden_size) #BERT
        
        
        self.embedding2 = nn.Linear(in2, in2) 
        self.unet1 = nn.Linear(in2, hidden_size)
        self.unet2 = nn.Linear(hidden_size, hidden_size)
        self.unet3 = nn.Linear(hidden_size,hidden_size)
        
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out)
        self.relu = nn.ReLU()


    def forward(self, x1, x2):

        #For BERT
        x1 = self.embedding1(x1)
        x1 = self.relu(x1)
                
        
        #For other modalities
        x2 = self.embedding2(x2)
        x2 = self.relu(x2)
        
        x2 = self.unet1(x2)
        x2 = self.relu(x2)
        
        x2 = self.unet2(x2)
        x2 = self.relu(x2)
        
        x2 = self.unet3(x2)
        x2 = self.relu(x2)

        
        z = torch.cat((x1, x2), dim=1)
        z = self.fc1(z)   
        z = self.relu(z)
        z = self.fc2(z)
        z = self.relu(z)
        
        return z





class FineTunedBERT(nn.Module):

    def __init__(self, pool="mean", model_name= "bert-base-cased", bert_state_dict=None,
                 task_type = None, n_labels = None, drop_rate = None,
                 gene2vec_flag=True, gene2vec_hidden = 200, device ="cuda"):
        
        """
            task_type : regression or classification.
        
        """
      
        super(FineTunedBERT, self).__init__()

#         assert (task_type == 'unsupervised' and n_labels == 1) or (task_type == 'regression' and n_labels == 1) or (task_type == 'classification' and n_labels>1) or (task_type == 'multilabel' and n_labels>1), \
#             f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"  

#         assert gene2vec_flag is not None, f"gene2vec_flag cannot be None: {gene2vec_flag}"

        
        self.model_name = model_name
        self.pool = pool
        
        
        if "xlnet" in model_name:
            self.bert = XLNetModel.from_pretrained(model_name).to(device)
        
        else:
            if bert_state_dict: 
                self.bert = AutoModel.from_pretrained(model_name)
                self.bert.load_state_dict(bert_state_dict) #.to(device)
            else:
                self.bert = AutoModel.from_pretrained(model_name)#.to(device)
                
        
#         layers_to_train = ["encoder.layer.11", "encoder.layer.10", "encoder.layer.9",
#                            "encoder.layer.8","pooler"]
        for name, param in self.bert.named_parameters():

            if name.startswith("encoder.layer.11") or name.startswith("encoder.layer.10") or\
            name.startswith("encoder.layer.9") or name.startswith("pooler") or\
            name.startswith("encoder.layer.8") : 
                param.requires_grad = True
            else:
                param.requires_grad = False
    
        
        
        self.bert_hidden = self.bert.config.hidden_size
        self.gene2vecFusion = LateFusion(in1=self.bert_hidden,
                                         in2=gene2vec_hidden,
                                         hidden_size=500,
                                         out = self.bert_hidden)
        
        if task_type.lower() == "classification":
            
            assert  n_labels > 1, f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"
            self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, n_labels))
#             if gene2vec_flag:
#                 self.pipeline = nn.Sequential(
#                     nn.Linear(self.bert_hidden+gene2vec_hidden, n_labels)
#                 )
                
#             else:
#                 self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, n_labels))
    
        elif task_type.lower() == "multilabel":
            assert  n_labels > 1, f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"
            self.pipeline = nn.Sequential(
                    nn.Linear(self.bert_hidden, n_labels),
                    nn.Sigmoid()
                )

            
#             assert  n_labels > 1, f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"
            
#             if gene2vec_flag:
#                 self.pipeline = nn.Sequential(
#                     nn.Linear(self.bert_hidden+gene2vec_hidden, n_labels),
#                     nn.Sigmoid()
#                 )
                
#             else:
#                 self.pipeline = nn.Sequential(
#                     nn.Linear(self.bert_hidden, n_labels),
#                     nn.Sigmoid()
#                 )

        elif task_type.lower() == "regression":
            assert  n_labels == 1, f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"
            self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))
            

#             if gene2vec_flag:
#                 self.pipeline = nn.Sequential(
#                 nn.Linear(self.bert_hidden+gene2vec_hidden, 1))
                
#             else:            
#                 self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))
        
        elif task_type.lower() =="interaction":
            if gene2vec_flag:
                raise ValueError(f"gene2vec_flag must be False when task_type is set to {task_type}: {gene2vec_flag=}")
                
            else:      
                self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))
            
        
        elif task_type.lower() == "unsupervised":
            self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))
            #No need for an assert, labels will not be used during unsupervised.

#             if gene2vec_flag:
#                 raise ValueError(f"gene2vec_flag must be False when task_type is set to {task_type}: {gene2vec_flag=}")
                
#             else:      
#                 self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))

        else:
            raise ValueError(f"Key Error task_type : {task_type} ")
          
    def forward(self, input_ids_, attention_mask_, gene2vec=None):
        
        
        # retrieving the hidden state embeddings
        if "xlnet" in self.model_name:
            output = self.bert(input_ids = input_ids_,
                               attention_mask=attention_mask_)

            hiddenState, ClsPooled = output.last_hidden_state, output.last_hidden_state[:,0, :]

        else:
            hiddenState, ClsPooled = self.bert(input_ids = input_ids_,
                                               attention_mask=attention_mask_).values()

        
        # perform pooling on the hidden state embeddings
        if self.pool.lower() == "max":
            embeddings = self.max_pooling(hiddenState, attention_mask_)
            
        elif self.pool.lower() == "cls":
            embeddings = ClsPooled
                
        elif self.pool.lower() == "mean":
            embeddings = self.mean_pooling(hiddenState, attention_mask_)

        else:
            raise ValueError('Pooling value error.')
        
        
        if gene2vec is not None:
            embeddings = self.gene2vecFusion(embeddings, gene2vec)
            #embeddings = torch.cat((embeddings, gene2vec), dim=1)
      

        return embeddings, hiddenState, self.pipeline(embeddings)

    def max_pooling(self, hidden_state, attention_mask):
        
        #CLS: First element of model_output contains all token embeddings
        token_embeddings = hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        
        pooled_embeddings = torch.max(token_embeddings, 1)[0]
        return pooled_embeddings
    
    def mean_pooling (self, hidden_state, attention_mask):
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9) 
        
        return pooled_embeddings
