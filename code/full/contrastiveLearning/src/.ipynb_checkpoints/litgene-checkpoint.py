import torch
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast, XLNetModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import collections


class LateFusion(nn.Module):
    def __init__(self, in1=768, in2=200, hidden_size=500, out = 768):
        super(Gene2VecFusionModel, self).__init__()
        
        self.embedding1 = nn.Linear(in1, hidden_size) #BERT
        
        
        self.embedding2 = nn.Linear(in2, in2) 
        self.unet1 = nn.Linear(in2, hidden_size)
        self.unet2 = nn.Linear(hidden_size, hidden_size)
        self.unet3 = nn.Linear(hidden_size,hidden_size)
        
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out)
        self.relu = nn.ReLU()


    def forward(self, x1, x2):
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

class FusionModel(nn.Module):
    def __init__(self, input_size=768, hidden_size=500, n_labels=1):
        super(FusionModel, self).__init__()
        
        self.embedding1 = nn.Linear(input_size, hidden_size)
        self.embedding2 = nn.Linear(input_size, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_labels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):

        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)
        
        z = torch.cat((x1, x2), dim=1)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        z = self.sigmoid(z)

        return z

class Gene2VecFusionModel(nn.Module):
    def __init__(self, in1=768, in2=200, hidden_size=500, out = 768):
        super(Gene2VecFusionModel, self).__init__()
        
        self.embedding1 = nn.Linear(in1, hidden_size)
        self.embedding2 = nn.Linear(in2, hidden_size)
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out)
        self.relu = nn.ReLU()


    def forward(self, x1, x2):

        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)
        
        z = torch.cat((x1, x2), dim=1)
        z = self.fc1(z)
        z = self.relu(z)
        z = self.fc2(z)
        z = self.relu(z)
        
        return z


class FineTunedBERT(nn.Module):

    def __init__(self, pool="mean", model_name= "bert-base-cased", bert_state_dict=None,
                 task_type = None, n_labels = None, drop_rate = None,
                 gene2vec_flag=None, gene2vec_hidden = 200, device ="cuda"):
        
        """
            task_type : regression or classification.
        
        """
      
        super(FineTunedBERT, self).__init__()
        
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
                
        for name, param in self.bert.named_parameters():

            if name.startswith("encoder.layer.11") or name.startswith("encoder.layer.10") or\
            name.startswith("encoder.layer.9") or name.startswith("pooler") or\
            name.startswith("encoder.layer.8") : 
                param.requires_grad = True
            else:
                param.requires_grad = False
    
        
        
        self.bert_hidden = self.bert.config.hidden_size
        self.gene2vecFusion = Gene2VecFusionModel(in1=self.bert_hidden,
                                                  in2=gene2vec_hidden,
                                                  hidden_size=500,
                                                  out = self.bert_hidden)
        
        if task_type.lower() == "classification":
            
            assert  n_labels > 1, f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"
            self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, n_labels))
    
        elif task_type.lower() == "multilabel":
            assert  n_labels > 1, f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"
            self.pipeline = nn.Sequential(
                    nn.Linear(self.bert_hidden, n_labels),
                    nn.Sigmoid()
                )


        elif task_type.lower() == "regression":
            assert  n_labels == 1, f"Invalid combination of task_type and n_labels: {task_type} and {n_labels}"
            self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))
            
        
        elif task_type.lower() =="interaction":
            if gene2vec_flag:
                raise ValueError(f"gene2vec_flag must be False when task_type is set to {task_type}: {gene2vec_flag=}")
                
            else:      
                self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 1))
            
        # Note: changed output shape to 2 (from 1)
        elif task_type.lower() == "unsupervised":
            self.pipeline = nn.Sequential(nn.Linear(self.bert_hidden, 2))

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
            
        elif self.pool.lower() == "sum":
            embeddings = self.sum_pooling(hiddenState, attention_mask_)
  
        else:
            raise ValueError('Pooling value error.')
        
        
        if gene2vec is not None:
            embeddings = self.gene2vecFusion(embeddings, gene2vec)
      
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
    
    def sum_pooling(self, hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
        pooled_embeddings = torch.sum(hidden_state * input_mask_expanded, 1)

        return pooled_embeddings
    
def getEmbeddings(text,
                  model = None,
                  model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                  max_length=512,
                  batch_size=1000,
                  gene2vec_flag=False,
                  gene2vec_hidden=200,
                  return_preds =False,
                  pool ="mean"): 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    if isinstance(model, FineTunedBERT):
        print("Loading a pretrained model ...")
        
        model_name = model.model_name            
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
        
#         print (model)
    
    elif isinstance(model, collections.OrderedDict):
        print("Loading a pretrained model from a state dictionary ...")
        
        state_dict = model.copy() 
        
        model = FineTunedBERT(pool= pool,model_name = model_name,
                              gene2vec_flag= gene2vec_flag,
                              gene2vec_hidden = gene2vec_hidden,
                              task_type="unsupervised",
                              n_labels = 1,
                              device = device).to(device)

#         model = nn.DataParallel(model)
        
        for s in model.state_dict().keys():
            if s not in state_dict.keys():
                print(s)
        
        model.load_state_dict(state_dict)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)
    else:
        print("Creating a new pretrained model ...")
        
        #BERT-base embeddings
        model = FineTunedBERT(pool= pool,
                              model_name=model_name,
                              task_type="unsupervised",
                              gene2vec_flag = False,
                              n_labels = 1).to(device)
  
        model = nn.DataParallel(model)
        tokenizer = BertTokenizerFast.from_pretrained(model_name)


    
    print("Tokenization ...")
    tokens = tokenizer.batch_encode_plus(text, max_length = max_length,
                                         padding="max_length",truncation=True,
                                         return_tensors="pt")
    
    
    dataset = TensorDataset(tokens["input_ids"] , tokens["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Tokenization Done.")
    
    print("Get Embeddings ...")
    
    model = model.to(device)
    embeddings=[]
    preds = []
    
    model.eval()
    for batch_input_ids, batch_attention_mask in tqdm(dataloader):
        with torch.no_grad():
            pooled_embeddings, _, p = model(batch_input_ids.to(device) ,
                                            batch_attention_mask.to(device))
            embeddings.append(pooled_embeddings)
            preds.append(p)
    
    
    concat_embeddings = torch.cat(embeddings, dim=0).cpu().detach().numpy()
    preds = torch.cat(preds, dim=0).cpu().detach().numpy()
    
    print(concat_embeddings.shape)
    print(preds.shape)
    
    if return_preds:
        return concat_embeddings, preds
        
    return concat_embeddings

def getEmbeddingsWithGene2Vec(dataloader, model, return_preds=True):
    device_opt = "cuda:0"
    device = torch.device( device_opt if torch.cuda.is_available() else "cpu")
        
    embeddings=[]
    preds = []
    model.eval()
    for batch in tqdm(dataloader):
        with torch.no_grad():
            batch_inputs_a, batch_masks_a, gene2vec_a  =  batch[0].to(device) , batch[1].to(device), batch[2].to(device)
            pooled_embeddings, _ , p= model(batch_inputs_a, batch_masks_a, gene2vec =gene2vec_a)
            embeddings.append(pooled_embeddings)
            preds.append(p)
    
    
    concat_embeddings = torch.cat(embeddings, dim=0).cpu().detach().numpy()
    preds = torch.cat(preds, dim=0).cpu().detach().numpy()
    
    print(concat_embeddings.shape)
    print(preds.shape)
    
    if return_preds:
        return concat_embeddings,preds
    return concat_embeddings
