from transformers import BertModel, BertTokenizer
from transformers import AdamW
import torch
from torch import nn
from torch.nn.functional import nll_loss
from torch.utils.data import Dataset, DataLoader
from utils import read_data, nodes_get_f1, edges_get_f1, sq_read_data
from torch.nn.functional import mse_loss
from torch.nn.functional import one_hot
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
from tabulate import tabulate
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from torchcrf import CRF
import sys
from transformers import DistilBertTokenizer, DistilBertModel
import logging
logging.basicConfig(level=logging.DEBUG)


class MultiDepthNodeEdgeDetector(torch.nn.Module):
    '''
    Neural Network architecture to extract named-entity and relation from a sentence.
    It leverages BERT for embeddings and has specialized layers to detect node start, node end, and edge span.
    '''
    def __init__(self, bert, tokenizer, dropout=0.5, clip_len=True, **kw):
        super().__init__(**kw)
        
        # Pre-trained BERT model for embeddings
        self.bert = bert
        
        # Number of last hidden states to concatenate from BERT
        self.concat_last_n = 3
        
        # Dimension size based on BERT's hidden size
        dim = self.bert.config.hidden_size
        
        # Linear layer for node start detection
        self.nodestart = torch.nn.Linear(self.concat_last_n * self.bert.config.hidden_size, 1)
        
        # Linear layer for node end detection
        self.nodeend = torch.nn.Linear(self.concat_last_n * self.bert.config.hidden_size, 1)
        
        # Linear layer for edge span detection
        self.edgespan = torch.nn.Linear(self.concat_last_n * self.bert.config.hidden_size, 1)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # A boolean to check if sequence lengths should be clipped
        self.clip_len = clip_len

        # Tokenizer for potential preprocessing needs
        self.tokenizer = tokenizer

    def forward(self, x):  # x: (batsize, seqlen) ints
        # Get the batch size and sequence size
        batch_size, seq_size = x.size()
        
        # Create a mask for non-padding tokens
        mask = (x != 0).long()
        
        # If clip_len is True, reduce the sequence length to max non-padding length
        if self.clip_len:
            maxlen = mask.sum(1).max().item()
            maxlen = min(x.size(1), maxlen + 1)
            mask = mask[:, :maxlen]
            x = x[:, :maxlen]
        
        # Get BERT outputs
        bert_outputs = self.bert(x, attention_mask=mask, output_hidden_states=True)
        
        # Concatenate the last 'concat_last_n' hidden states
        hidden = torch.cat(bert_outputs.hidden_states[-self.concat_last_n:], dim=-1)
        
        # Apply dropout
        a = self.dropout(hidden)
        
        # Compute logits for node start, node end, and edge span
        logits_node_start = self.nodestart(a)
        logits_node_end = self.nodeend(a)
        logits_edge_span = self.edgespan(a)
        
        # Combine all logits
        logits = torch.cat([
            logits_node_start.transpose(1, 2),
            logits_node_end.transpose(1, 2),
            logits_edge_span.transpose(1, 2)
        ], 1)
        
        return logits


class NodeEdgeDetector(torch.nn.Module):
    '''
    NodeEdgeDetector is a neural network architecture designed to recognize entity and relation 
    spans in a question posed in natural language.
    
    This model is built on top of a pre-trained BERT model and introduces several additional 
    linear layers to predict start and end positions of entity nodes and edge spans.
    '''
    
    def __init__(self, bert, tokenizer, dropout=0.5, clip_len=True, **kw):
        """
        Initialize the NodeEdgeDetector model.
        
        :param bert: Pre-trained BERT model
        :param tokenizer: BERT tokenizer for tokenization
        :param dropout: Dropout rate for regularization
        :param clip_len: Flag to determine if sequence length should be clipped
        """
        super().__init__(**kw)
        self.bert = bert  # BERT model
        dim = self.bert.config.hidden_size  # Hidden size of BERT model
        
        # Linear layers to predict the start and end positions of entity nodes
        self.nodestart = torch.nn.Linear(dim, 1)
        self.nodeend = torch.nn.Linear(dim, 1)
        
        # Linear layer to predict the span of edges
        self.edgespan = torch.nn.Linear(dim, 1)
        
        # Dropout layer for regularization
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # Flag to clip input sequence length
        self.clip_len = clip_len

        # BERT tokenizer
        self.tokenizer = tokenizer

    def forward(self, x):  # x: (batsize, seqlen) ints
        """
        Forward pass of the model.
        
        :param x: Input tensor of token IDs
        :return: logits: Model output logits for node start, node end, and edge span
        """
        # Create an attention mask to identify non-zero tokens (used for variable-length sequences)
        mask = (x != 0).long()
        
        # Optionally clip the sequence length to the maximum actual sequence length in the batch
        if self.clip_len:
            maxlen = mask.sum(1).max().item()
            maxlen = min(x.size(1), maxlen + 1)
            mask = mask[:, :maxlen]
            x = x[:, :maxlen]
        
        # Obtain outputs from BERT model
        bert_outputs = self.bert(x, attention_mask=mask, output_hidden_states=False)
        lhs = bert_outputs.last_hidden_state  # Extract the last hidden state
        
        # Apply dropout
        a = self.dropout(lhs)
        
        # Predict the start and end positions for nodes and the span for edges
        logits_node_start = self.nodestart(lhs)
        logits_node_end = self.nodeend(lhs)
        logits_edge_span = self.edgespan(lhs)
        
        # Concatenate the logits for each prediction
        logits = torch.cat([logits_node_start.transpose(1, 2), 
                            logits_node_end.transpose(1, 2), 
                            logits_edge_span.transpose(1, 2)], 1)
        
        return logits


class BertCNN(torch.nn.Module):
    '''
    Neural Network architecture that combines BERT embeddings with Convolutional Neural Networks (CNN) 
    to extract named-entity and relation from a sentence.
    '''
    def __init__(self, bert, tokenizer, dropout=0.5, clip_len=True, **kw):
        super().__init__(**kw)
        
        # Pre-trained BERT model for embeddings
        self.bert = bert
        
        # Dimension size based on BERT's hidden size
        dim = self.bert.config.hidden_size
        
        # Linear layers for node start, node end, and edge span detection
        self.nodestart = torch.nn.Linear(160, 35)
        self.nodeend = torch.nn.Linear(160, 35)
        self.edgespan = torch.nn.Linear(160, 35)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # A boolean to check if sequence lengths should be clipped
        self.clip_len = clip_len
        
        # Tokenizer for potential preprocessing needs
        self.tokenizer = tokenizer
        
        # Define filter sizes and number of filters for the CNN
        filter_sizes = [1,2,3,4,5]
        num_filters = 32
        embed_size = 768
        
        # Convolutional layers
        self.convs1 = nn.ModuleList([nn.Conv2d(4, num_filters, (K, embed_size)) for K in filter_sizes])
        
    def forward(self, x):  # x: (batsize, seqlen) ints
        # Create a mask for non-padding tokens
        mask = (x != 0).long()
        
        # If clip_len is True, reduce the sequence length to max non-padding length
        if self.clip_len:
            maxlen = mask.sum(1).max().item()
            maxlen = min(x.size(1), maxlen + 1)
            mask = mask[:, :maxlen]
            x = x[:, :maxlen]
        
        # Get BERT outputs for the last 4 hidden states
        x = self.bert(x, attention_mask=mask, output_hidden_states=True)[2][-4:]
        
        # Stack the 4 hidden states along a new dimension
        x = torch.stack(x, dim=1)
        
        # Pass the stacked hidden states through the convolutional layers
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1] 
        
        # Apply max pooling
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  
        
        # Concatenate the output from all convolutional layers
        x = torch.cat(x, 1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Compute logits for node start, node end, and edge span
        logits_node_start = torch.unsqueeze(self.nodestart(x), 1)
        logits_node_end = torch.unsqueeze(self.nodeend(x), 1)
        logits_edge_span = torch.unsqueeze(self.edgespan(x), 1)
        
        # Combine all logits ensuring they are of the same length as input sequence
        b, l = x.size()  # Extract batch size and length for slicing
        logits = torch.cat([
            logits_node_start[:, :, :l], 
            logits_node_end[:, :, :l], 
            logits_edge_span[:, :, :l]
        ], 1)
        
        return logits


class BertLSTMCRF(torch.nn.Module):
    '''
    Neural Network architecture that combines BERT embeddings with LSTM and Conditional Random Fields (CRF) 
    to extract named-entity and relation from a sentence.
    '''
    def __init__(self, bert, tokenizer, dropout=0.5, clip_len=True, **kw):
        super().__init__(**kw)
        
        # Pre-trained BERT model for embeddings
        self.bert = bert
        
        # Dimension size based on BERT's hidden size
        dim = self.bert.config.hidden_size
        
        # Linear layers for node start, node end, and edge span detection
        self.nodestart = torch.nn.Linear(dim, 1)
        self.nodeend = torch.nn.Linear(dim, 1)
        self.edgespan = torch.nn.Linear(dim, 1)
        
        # Dropout layer
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # A boolean to check if sequence lengths should be clipped
        self.clip_len = clip_len
        
        # Tokenizer for potential preprocessing needs
        self.tokenizer = tokenizer
        
        # CRF layers for entity and relation extraction
        self.crf_entity = CRF(2, batch_first=True)
        self.crf_relation = CRF(2, batch_first=True)
        
        # LSTM layer for sequence modeling
        self.lstm = torch.nn.LSTM(
            input_size=dim,
            hidden_size=100,
            num_layers=2,
            dropout=0.2,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):  # x: (batsize, seqlen) ints
        # Create a mask for non-padding tokens
        mask = (x != 0).long()
        
        # If clip_len is True, reduce the sequence length to max non-padding length
        if self.clip_len:
            maxlen = mask.sum(1).max().item()
            maxlen = min(x.size(1), maxlen + 1)
            mask = mask[:, :maxlen]
            x = x[:, :maxlen]
        
        # Get BERT outputs for the last hidden state
        bert_outputs = self.bert(x, attention_mask=mask, output_hidden_states=False)
        
        # Extract the last hidden state
        lhs = bert_outputs.last_hidden_state
        
        # Apply dropout
        a = self.dropout(lhs)
        
        # Get LSTM outputs
        lstm_out, *_ = self.lstm(a)
        
        # Compute logits for node start, node end, and edge span
        logits_node_start = self.nodestart(lhs)
        logits_node_end = self.nodeend(lhs)
        logits_edge_span = self.edgespan(lhs)
        
        # Combine all logits
        logits = torch.cat([
            logits_node_start.transpose(1, 2),
            logits_node_end.transpose(1, 2), 
            logits_edge_span.transpose(1, 2)
        ], 1)
        
        return logits


class BordersDataset(Dataset):
	'''
		Convert Data to proper Tensor dataset
	'''
	def __init__(self, data):
		# convert into tensors
		self.tokens_matrix = torch.from_numpy(data[0]).long()
		self.borders = torch.from_numpy(data[1]).long()
		self.n_samples = data[0].shape[0]

	def __getitem__(self, index):
		# returns specific item
		return self.tokens_matrix[index], self.borders[index] 
	def __len__(self):
		return self.n_samples
		# returns dataset length



class TrainingLoop:
	'''
	Everything related to model training
	'''
	def __init__(self, model, optimizer, freezeemb=True, 
				 epochs=6, save_path='./models/', **kw):
		self.model = model
		params = []
		for paramname, param in self.model.named_parameters():
			if paramname.startswith("bert.embeddings.word_embeddings"):
				if not freezeemb:
					params.append(param)
			else:
				params.append(param)
		self.optimizer = optimizer(params, **kw)
		self.epochs = epochs
		self.save_path = save_path
		self.predicts = None
	def train(self, dataloader, eval_dataloader, loss_function):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model.to(device)
		
		for epoch in range(self.epochs):
			self.model.train()
			losses = []

			for _, batch in enumerate(tqdm(dataloader, desc=f"Train Epoch Number {epoch+1}")):
				self.model.zero_grad()
				X, y = batch
				X = X.to(device); y = y.to(device)
				logits = self.model(X) 
				nodes_onehot = one_hot(y[:, :2], num_classes=logits.size()[-1]).float()
				
				maxlen = logits.size()[-1]
				actual = torch.cat((nodes_onehot, torch.unsqueeze(y[:, 2:], 1)[:, :, :maxlen]), 1)
				

				loss = loss_function(logits, actual, reduction='sum')
				losses.append(loss)
				loss.backward()
				self.optimizer.step()
			logging.info(f'Epoch number: {epoch+1} Train Loss is equal: {sum(losses)/len(losses)}') 
			self.eval(eval_dataloader, loss_function, epoch, device)


	def eval(self, dataloader, loss_function, epoch, device):
		self.model.eval()
		losses = []
		for _, batch in enumerate(tqdm(dataloader, desc=f"Eval Epoch Number {epoch+1}")):
			with torch.no_grad():
				X, y = batch
				X = X.to(device); y = y.to(device)
				logits = self.model(X) 
				nodes_onehot = one_hot(y[:, :2], num_classes=logits.size()[-1]).float()
				
				maxlen = logits.size()[-1]
				actual = torch.cat((nodes_onehot, torch.unsqueeze(y[:, 2:], 1)[:, :, :maxlen]), 1)
				

				loss = loss_function(logits, actual, reduction='sum')
				losses.append(loss)
		logging.info(f'Epoch number: {epoch+1} Eval Loss is equal: {sum(losses)/len(losses)}')

	def predict(self, dataloader, device, evaluate=True):
		self.model.eval()
		predicts = []
		for _, batch in enumerate(tqdm(dataloader, desc=f"Predicting ...")):
			with torch.no_grad():
				X, _ = batch
				X = X.to(device)
				logits = self.model(X)
				nodes_borders = torch.argmax(logits[:, :2], dim=2).cpu().detach().numpy().tolist()
				edges_spans = np.where(logits[:, 2].cpu().detach().numpy() > 0.5, 1, 0)
				
				[predicts.append((node_borders, edge_spans)) for node_borders, edge_spans in zip(nodes_borders, edges_spans)]
		self.predicts = predicts
		if evaluate:
			node_goldens = []
			for batch in dataloader:
				_, y = batch
				nodes_borders = y[:, :2].cpu().detach().numpy().tolist()
				[node_goldens.append(item) for item in nodes_borders]
			gold_nodes_border = np.array(node_goldens)
			pred_nodes_border = np.array([item[0] for item in self.predicts])
			nodes_get_f1(pred_nodes_border, gold_nodes_border)
			
			edge_goldens = []
			for batch in dataloader:
				_, y = batch
				edges_spans = y[:, 2:].cpu().detach().numpy().tolist()
				[edge_goldens.append(item) for item in edges_spans]
			gold_edges_span = np.array(edge_goldens)
			pred_edges_span = [item[1].tolist()+[0 for _ in range(35-len(item[1]))] for item in self.predicts]
			pred_edges_span = np.asarray(pred_edges_span)
			# print(gold_edges_span.shape, pred_edges_span.shape)
			edges_get_f1(pred_edges_span, gold_edges_span)

	
	def save(self, save_path='../models/node_edge_bert.pt'):
		torch.save(self.model, save_path)
	
	def load(self, save_path='../models/node_edge_bert.pt'):
		self.model = torch.load(save_path)

	def readable_predict(self, device, _input='Where was Bill Gates Born?', print_result=True):
		addspecialtokens = lambda string:f'[CLS] {string} [SEP]'
		wordstoberttokens = lambda string:self.model.tokenizer.tokenize(string)
		berttokenstoids = lambda tokens:self.model.tokenizer.convert_tokens_to_ids(tokens)
		input_token_ids = berttokenstoids(wordstoberttokens(addspecialtokens(_input)))
		input_tensors = torch.tensor([input_token_ids]).long()
		input_tensors = input_tensors.to(device)
		with torch.no_grad():
			logits = self.model(input_tensors)
		nodes_borders = torch.argmax(logits[:, :2], dim=2).cpu().detach().numpy().tolist()
		edges_spans = np.where(logits[:, 2].cpu().detach().numpy() > 0.5, 1, 0)

		node = self.model.tokenizer.convert_ids_to_tokens(input_token_ids[nodes_borders[0][0]:nodes_borders[0][1]])
		edge = self.model.tokenizer.convert_ids_to_tokens(np.array(input_token_ids)[edges_spans[0]==1])
		if print_result:
			data = [[_input, node, edge]]
			print(tabulate(data, headers=["Question", "Node", "Edge"]))
		else:
			return node, edge
	def readable_predict_article(self, device, _input='Where was Bill Gates Born?', print_result=True):
    ### getting node and edge
		addspecialtokens = lambda string:f'[CLS] {string} [SEP]'
		wordstoberttokens = lambda string:self.model.tokenizer.tokenize(string)
		berttokenstoids = lambda tokens:self.model.tokenizer.convert_tokens_to_ids(tokens)
		input_token_ids = berttokenstoids(wordstoberttokens(addspecialtokens(_input)))
		input_tensors = torch.tensor([input_token_ids]).long()
		input_tensors = input_tensors.to(device)
		with torch.no_grad():
			logits = self.model(input_tensors)
		nodes_borders = torch.argmax(logits[:, :2], dim=2).cpu().detach().numpy().tolist()
		edges_spans = np.where(logits[:, 2].cpu().detach().numpy() > 0.5, 1, 0)

		node = self.model.tokenizer.convert_ids_to_tokens(input_token_ids[nodes_borders[0][0]:nodes_borders[0][1]])
		edge = self.model.tokenizer.convert_ids_to_tokens(np.array(input_token_ids)[edges_spans[0]==1])
		if print_result:
			data = [[_input, node, edge]]
			print(tabulate(data, headers=["Question", "Node", "Edge"]))
		else:
			return wordstoberttokens, berttokenstoids, input_token_ids, nodes_borders, edges_spans, node, edge

if __name__=='__main__':
	model_type = sys.argv[1]
	mapping = {'MultiDepthNodeEdgeDetector':MultiDepthNodeEdgeDetector,
              'BertLSTMCRF':BertLSTMCRF, 
              'BertCNN':BertCNN,
              'NodeEdgeDetector':NodeEdgeDetector
            }
	cross_validation = eval(sys.argv[2])
	
	if cross_validation == True:
		train, valid, test = read_data()
		X = np.vstack((train[0], valid[0], test[0]))
		y = np.vstack((train[1], valid[1], test[1]))
		kf = KFold(n_splits=4, random_state=99, shuffle=True)
		fold = 1
		for train_index, test_index in kf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=99)
			train = (X_train, y_train)
			valid = (X_valid, y_valid)
			test = (X_test, y_test)
			logging.info(f'\n\n############# Fold Number {fold} #############\n\n')
			fold+=1
			bert = BertModel.from_pretrained("bert-base-uncased")
			tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
			node_edge_detector = mapping[model_type](bert, tokenizer, dropout=torch.tensor(0.5))
			optimizer = AdamW
			kw = {'lr':0.0002, 'weight_decay':0.1}
			tl = TrainingLoop(node_edge_detector, optimizer, True, **kw)
			
			train_dataset = BordersDataset(train)
			train_dataloader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True, pin_memory=True)
			valid_dataset = BordersDataset(valid)
			valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=200, shuffle=False, pin_memory=True)
			test_dataset = BordersDataset(test)
			test_dataloader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, pin_memory=True)
			logging.info(f'Train Dataset Contains {len(train_dataset)} Samples.')
			logging.info(f'Valid Dataset Contains {len(valid_dataset)} Samples.')
			logging.info(f'Test Dataset Contains {len(test_dataset)} Samples.')
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			loss = mse_loss
			tl.train(train_dataloader, valid_dataloader, loss)
			tl.save()
			##################################################
			tl.load()
			tl.predict(test_dataloader, device)
			##################################################
			tl.readable_predict(device, print_result=True)

	else:
		# train, valid, test = sq_read_data('train'), sq_read_data('valid'), sq_read_data('test')
		train, valid, test = read_data()
		bert = BertModel.from_pretrained("bert-base-uncased")
		tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
		node_edge_detector = mapping[model_type](bert, tokenizer, dropout=torch.tensor(0.5))
		optimizer = AdamW
		kw = {'lr':0.0002, 'weight_decay':0.1}
		tl = TrainingLoop(node_edge_detector, optimizer, True, **kw)
		
		train_dataset = BordersDataset(train)
		train_dataloader = DataLoader(dataset=train_dataset, batch_size=200, shuffle=True, pin_memory=True)
		valid_dataset = BordersDataset(valid)
		valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=200, shuffle=False, pin_memory=True)
		test_dataset = BordersDataset(test)
		test_dataloader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=False, pin_memory=True)
		
		logging.info(f'Train Dataset Contains {len(train_dataset)} Samples.')
		logging.info(f'Valid Dataset Contains {len(valid_dataset)} Samples.')
		logging.info(f'Test Dataset Contains {len(test_dataset)} Samples.')

		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		loss = mse_loss
		tl.train(train_dataloader, valid_dataloader, loss)
		tl.save()
		##################################################
		tl.load()
		tl.predict(test_dataloader, device)
		##################################################
		tl.readable_predict(device, print_result=True)
		
