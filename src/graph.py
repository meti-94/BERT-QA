import pandas as pd
import networkx as nx
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import get_tf_idf_query_similarity
from sklearn.metrics.pairwise import cosine_similarity
from pattern.en import conjugate, lemma, lexeme,PRESENT,SG,PAST
import sys
import re
import pyterrier as pt
pt.init()
checkpoint="http://www.dcs.gla.ac.uk/~craigm/colbert.dnn.zip"
import pyterrier_colbert.indexing
import logging
logging.basicConfig(level=logging.WARNING)
#### pattern python>=3.7 compatibility problem
def pattern_stopiteration_workaround():
    try:
        print(lexeme('gave'))
    except:
        pass
pattern_stopiteration_workaround()


'''
KnowledgeBase Utililies / Managment
indexing .... 
Searchin with tfidf and Cosign Similarity 
'''

class ReverbKnowledgeBaseGraph:
	def __init__(self, path='../data/reverb_wikipedia_tuples-1.1.txt'):
		super().__init__()
		df = pd.read_csv(path, sep='\t', header=None)
		reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
		df.columns = reverb_columns_name
		df = df.dropna()
		df = df.drop_duplicates()
		KBG = nx.MultiGraph()
		for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Reading Graph ...'):
			KBG.add_nodes_from([(row['arg1'], {'alias':row['narg1'], 'reverb_line_no':index})])
			KBG.add_nodes_from([(row['arg2'], {'alias':row['narg2'], 'reverb_line_no':index})])
			KBG.add_edges_from([(row['arg1'], row['arg2'], 
									{'nrel':row['nrel'], 'alias':row['rel'], 'csents':row['csents'], 'conf':row['conf'], 'ExID':row['ExID']})])
		self.KBG = KBG
		self.nodes = self.KBG.nodes
		self.edges = nx.get_edge_attributes(self.KBG,'alias')
		self.nodes_vectorizer = TfidfVectorizer()
		self.edges_vectorizer = TfidfVectorizer()
		self.nodes_tfidf = self.nodes_vectorizer.fit_transform(self.nodes)
		self.edges_tfidf = self.edges_vectorizer.fit_transform(self.edges.values())


	def nodesquery(self, search_phrase, cutoff=80):
		candidates = process.extractBests(search_phrase, self.nodes, score_cutoff=cutoff, limit=len(self.nodes)-1)
		return candidates

	def edgesquery(self, search_phrase, cutoff=80):
		candidates = process.extractBests(search_phrase, self.edges, score_cutoff=cutoff, limit=len(self.edges)-1)
		return candidates

	def query(self, node='Bill Gates', edge='Born'):
		nodes = self.nodesquery(node)
		edges = self.edgesquery(edge)
		candidates = []
		for nd in nodes:
			for ed in edges:
				if ed[-1][0]==nd[-1]:
					candidates.append((nd[0]['reverb_line_no'], nd[-1], nd[1], ed[0], ed[1], ed[-1][1]))
				if ed[-1][1]==nd[-1]:
					candidates.append((nd[0]['reverb_line_no'], nd[-1], nd[1], ed[0], ed[1], ed[-1][0]))

		return candidates

	def tfidf_nodes_query(self, search_phrase, cutoff=80):
		similarities = get_tf_idf_query_similarity(self.nodes_vectorizer, self.nodes_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.nodes, similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)}

		return sorted_ranks

	def tfidf_edges_query(self, search_phrase, cutoff=80):
		similarities = get_tf_idf_query_similarity(self.edges_vectorizer, self.edges_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.edges.keys(), similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)}
		return sorted_ranks

	def tfidf_query(self, node='Bill Gates', edge='Born'):
		nodes = self.nodesquery(node)
		print(nodes)
		edges = self.edgesquery(edge)

class ReverbKnowledgeBaseNN:
	def __init__(self, path='../data/reverb_wikipedia_tuples-1.1.txt'):
		super().__init__()
		df = pd.read_csv(path, sep='\t', header=None)
		df = df[:5000]
		reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
		df.columns = reverb_columns_name
		df = df.dropna()
		df = df.drop_duplicates()
		df['text'] = df.apply(lambda row:' '.join([row['arg1'], row['rel'], row['arg2']]).strip(), axis=1)
		df['docno'] = (df.index).astype(str)
		indexer = pyterrier_colbert.indexing.ColBERTIndexer(checkpoint, "/content", "colbertindex", chunksize=3)
		indexer.index(df[['text', 'docno']].to_dict(orient="records"))
		
		pyterrier_colbert_factory = indexer.ranking_factory()
		e2e = pyterrier_colbert_factory.end_to_end()
		self.e2e = e2e


	def NN_nodes_query(self, search_phrase, cutoff=2500):

		q = re.sub(r'[^\w]', ' ', search_phrase)
		ret = (self.e2e % cutoff).search(q)[['score', 'docno']]
		return ret[:cutoff]

	def NN_edge_query(self, search_phrase, cutoff=2500):

		q = re.sub(r'[^\w]', ' ', search_phrase)
		ret = (self.e2e % cutoff).search(q)[['score', 'docno']]
		return ret

	def query(self, node='Bill Gates', edge='Born'):
		nodes_df = self.NN_nodes_query(node)
		edges_df = self.NN_nodes_query(edge)
		candidates = nodes_df.merge(edges_df, left_on='docno', right_on='docno', how='inner')
		candidates['score'] = candidates['score_x']+candidates['score_y']
		candidates.sort_values(by='score', ascending=False)
		return candidates[:min(10, len(candidates))]['docno'].astype(int).to_list()
		
	

class ReverbKnowledgeBase:
	def __init__(self, path='../data/reverb_wikipedia_tuples-1.1.txt'):
		super().__init__()
		df = pd.read_csv(path, sep='\t', header=None)
		reverb_columns_name = ['ExID', 'arg1', 'rel', 'arg2', 'narg1', 'nrel', 'narg2', 'csents', 'conf', 'urls']
		df.columns = reverb_columns_name
		df = df.dropna()
		df = df.drop_duplicates()
		self.KB = df
		self.is_facts = self.KB[(self.KB.rel.apply(lambda rg:rg.find('is ')!=-1))|(self.KB.rel.apply(lambda rg:rg.find('Is ')!=-1))]
		self.nodes = self.KB['arg1'].to_list()+self.KB['arg2'].to_list()
		self.edges = self.KB['rel'].to_list()
		self.nodes_vectorizer = TfidfVectorizer()
		self.edges_vectorizer = TfidfVectorizer()
		self.nodes_tfidf = self.nodes_vectorizer.fit_transform(self.nodes)
		self.edges_tfidf = self.edges_vectorizer.fit_transform(self.edges)
		self.relations = {}
		for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Indexing ...'):
			if row['rel'] in self.relations:
				self.relations[row['rel']].append((row['arg1'], index, row['conf']))
				self.relations[row['rel']].append((row['arg2'], index, row['conf']))
			else:
				self.relations[row['rel']] = [(row['arg1'], index, row['conf'])]
				self.relations[row['rel']].append((row['arg2'], index, row['conf']))
		


	def tfidf_nodes_query(self, search_phrase, cutoff=50):
		similarities = get_tf_idf_query_similarity(self.nodes_vectorizer, self.nodes_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.nodes, similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)[:min(len(ranks), cutoff)]}

		return sorted_ranks

	def tfidf_edges_query(self, search_phrase, cutoff=50):
		similarities = get_tf_idf_query_similarity(self.edges_vectorizer, self.edges_tfidf, search_phrase)
		ranks = {k:v for k,v in zip(self.edges, similarities)}
		sorted_ranks = {k: v for k, v in sorted(ranks.items(), key=lambda item:item[1], reverse=True)[:min(len(ranks), cutoff)]}
		return sorted_ranks

	def query(self, node='Bill Gates', edge='Born'):
		# print(edge)
		edge_list = edge.split()
		if len(edge_list)>=2 and edge_list[0]=='did':
			edge_list[1] = conjugate(verb=edge_list[1],tense=PAST)
			edge = ' '.join(edge_list[1:])
		else:
			edge = ' '.join(edge_list)
		# print(edge)
		if edge.strip()!='is':
			nodes = self.tfidf_nodes_query(node)
			edges = self.tfidf_edges_query(edge)
			pruned = []
			for node in nodes.keys():
				for edge in edges.keys():
					for item in self.relations[edge]:
						if item[0]==node:
							pruned.append((item[1], item[-1], nodes[node], edges[edge]))
			sorted_pruned = sorted(pruned, key=lambda x:x[2]+x[3], reverse=True)
			return sorted_pruned[:min(len(sorted_pruned), 100)]
		else:
			nodes = self.tfidf_nodes_query(node)
			arg1 = self.KB.loc[self.KB['arg1'].isin(nodes.keys())]
			arg2 = self.KB.loc[self.KB['arg2'].isin(nodes.keys())]
			# print(self.KB.loc[self.KB['arg2'].isin(nodes.keys())][:10])
			
			pruned = []
			for node, similarity in nodes.items():
				for idx, row in arg1.loc[arg1['arg1']==node].iterrows():
					temp1 = self.edges_vectorizer.transform([row['rel']])
					temp2 = self.edges_vectorizer.transform([edge])
					edge_similarity = cosine_similarity(temp1, temp2).flatten().item()
					pruned.append((idx, row['conf'], similarity, edge_similarity))
				for idx, row in arg2.loc[arg2['arg2']==node].iterrows():
					temp1 = self.edges_vectorizer.transform([row['rel']])
					temp2 = self.edges_vectorizer.transform([edge])
					edge_similarity = cosine_similarity(temp1, temp2).flatten().item()
					pruned.append((idx, row['conf'], similarity, edge_similarity))
			sorted_pruned = sorted(pruned, key=lambda x:x[2]+x[3], reverse=True)
			return sorted_pruned[:min(len(sorted_pruned), 100)]

if __name__=='__main__':
	RKBNN = ReverbKnowledgeBaseNN('/content/reverb_wikipedia_tuples-1.1.txt') #	'./sample_reverb_tuples.txt'
	# print(len(RKBG.nodes_vectorizer.vocabulary_), len(RKBG.edges_vectorizer.vocabulary_))
	# print(RKBG.tfidf_query(node='fishkind', edge='grew up in'))
	print(RKBNN.NN_query(node='abegg', edge='did die'))
