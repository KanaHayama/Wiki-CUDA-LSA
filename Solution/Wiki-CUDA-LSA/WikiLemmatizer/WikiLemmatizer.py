import xml
import subprocess
from WikiXmlHandler import WikiXmlHandler
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora
import argparse
import sys
import numpy as np

def preprocess_data(doc):
	"""
	Input  : docuemnt list
	Purpose: preprocess text (tokenize, removing stopwords, and stemming)
	Output : preprocessed text
	"""
	# initialize regex tokenizer
	tokenizer = RegexpTokenizer(r'\w+')
	# create English stop words list
	en_stop = set(stopwords.words('english'))
	# Create p_stemmer of class PorterStemmer
	p_stemmer = PorterStemmer()
	# list for tokenized documents in loop
	texts = []
	# loop through document list
	raw = doc.lower()
	tokens = tokenizer.tokenize(raw)
	# remove stop words from tokens
	stopped_tokens = [i for i in tokens if not i in en_stop]
	# stem tokens
	stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
	# add tokens to list
	texts.extend(stemmed_tokens)
	return texts

def prepare_corpus(docs_clean):
	"""
	Input  : clean document
	Purpose: create term dictionary of our courpus and Converting list of documents (corpus) into Document Term Matrix
	Output : term dictionary and Document Term Matrix
	"""
	# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
	dictionary = corpora.Dictionary(docs_clean)
	# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
	sparse = [dictionary.doc2bow(doc) for doc in docs_clean]
	# generate LDA model
	
	doc_term_freq_matrix = []
	numTerm = len(dictionary)
	for doc_terms in sparse:
		term_freq = [0 for _ in range(numTerm)]
		for id, count in doc_terms:
			term_freq[id] = count
		doc_term_freq_matrix.append(term_freq)
	return dictionary, doc_term_freq_matrix

def print_terms(dictionary, file=sys.stdout):
	print("Terms:", file=file)
	for id, term in dictionary.iteritems():
		print("{}:{}".format(id, term), file=file)

def print_doc_titles(titles, file=sys.stdout):
	print("Doc titles:", file=file)
	for id, title in titles:
		print("{}:{}".format(id, title), file=file)

def print_doc_term_freq_matrix(doc_term_freq_matrix, file=sys.stdout):
	print("Doc term freq matrix:", file=file)
	for doc_freq in doc_term_freq_matrix:
		print(" ".join(str(freq) for freq in doc_freq), file=file)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "For USC EE451 2019 spring home work.")
	parser.add_argument("-a", "--articles", type=int, default=None, help=".")
	parser.add_argument("-z", "--zipped", action="store_true", default=False, help="")
	parser.add_argument("-i", "--input", nargs='?', type=argparse.FileType('r', encoding='UTF-8'), default=sys.stdin, help="")
	parser.add_argument("-o", "--output", nargs='?', type=argparse.FileType('w', encoding='UTF-8'), default=sys.stdout, help="")
	args = parser.parse_args()

	# Object for handling xml
	handler = WikiXmlHandler()
	# Parsing object
	parser = xml.sax.make_parser()
	parser.setContentHandler(handler)
	# Iteratively process file
	input = subprocess.Popen(['bzcat'], stdin = args.input, stdout = subprocess.PIPE).stdout if args.zipped else args.input
	for line in input:
		parser.feed(line)
		if args.articles is not None and len(handler._pages) >= args.articles:
			break
	
	from datetime import datetime
	print(datetime.now())
	title_doc_pairs = handler._pages
	# print(title_doc_pairs)
	docs_clean = [preprocess_data(pair[1]) for pair in title_doc_pairs]
	term_dict, doc_term_freq_matrix = prepare_corpus(docs_clean)
	title_dict = [(i, title_doc_pairs[i][0]) for i in range(len(title_doc_pairs))]
	print("{} docs, {} terms\n".format(len(docs_clean), len(term_dict)), file=args.output)
	print_terms(term_dict, args.output)
	print_doc_titles(title_dict, args.output)
	print_doc_term_freq_matrix(doc_term_freq_matrix, args.output)
