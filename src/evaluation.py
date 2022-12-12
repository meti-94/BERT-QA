from train import *
from graph import *
import pandas as pd
from utils import get_hit
'''
Script to perform whole system Evaluation
'''


if __name__=='__main__':
	similarity_type = sys.argv[1]
	mapping = {'tfidf':ReverbKnowledgeBase,
              'NN':ReverbKnowledgeBaseNN, 
            }
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	bert = BertModel.from_pretrained("bert-base-uncased")
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
	node_edge_detector = NodeEdgeDetector(bert, tokenizer, dropout=torch.tensor(0.5))
	optimizer = AdamW
	kw = {'lr':0.0002, 'weight_decay':0.1}
	tl = TrainingLoop(node_edge_detector, optimizer, True, **kw)
	loss = mse_loss
	tl.load()

	RKBG = mapping[similarity_type]()

	test_df = pd.read_excel('../data/test.xlsx')
	actual = test_df['Reverb_no'].to_list()
	system_results = []
	for index, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
		node, edge = tl.readable_predict(device, _input=row['Question'], print_result=False)
		node = ' '.join(node); edge = ' '.join(edge)
		node = node.replace(' ##', ''); edge = edge.replace(' ##', '')
		if index%100==0:
			print("The Question: ", row['Question'].lower().split())
			print(f'Node: {node}, Edge: {edge}')
		temp = RKBG.query(node=node, edge=edge)
		# print('Sorted candidates: ', temp[:min(len(temp), 25)])
		# print('Actual line number: ', row['Reverb_no'])
		system_results.append(temp)
	print(get_hit(actual, system_results))

		
