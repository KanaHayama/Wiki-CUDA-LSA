
#include<iostream>
#include<fstream>
#include<vector>
#include<tfidf.cuh>
#include<svd.cuh>
#include<string>
#include<cassert>
#include"lsa.cuh"

using std::vector;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::get;

shared_ptr<float> read(const string filename, int & numDocs, int & numTerms, vector<string> & docTitles, vector<string> & terms) {
	auto f = std::ifstream(filename);
	assert(f.is_open());
	f >> numDocs >> numTerms;
	
	char buf[1024];
	f.getline(buf, 1024);
	for (auto i = 0; i < numDocs; i++) {
		f.getline(buf, 1024);
		docTitles.push_back(string(buf));
	}
	for (auto i = 0; i < numTerms; i++) {
		f.getline(buf, 1024);
		terms.push_back(string(buf));
	}
	auto term_freq_mat = shared_ptr<float>(new float[numDocs * numTerms]);
	for (auto i = 0; i < numDocs; i++) {
		for (auto j = 0; j < numTerms; j++) {
			int val;
			f >> val;
			term_freq_mat.get()[i * numTerms + j] = val;
		}
	}
	f.close();
	return term_freq_mat;
}

void printConcepts(const int numConcept, const int numTerms, const int numDocs, const std::vector<std::vector<std::tuple<int, float>>> topDocs, const std::vector<std::vector<std::tuple<int, float>>> topTerms, const vector<string> terms, const vector<string> docTitles) {
	for (int c = 0; c < numConcept; c++) {
		std::cout << "Concept " << c << ":" << std::endl;
		std::cout << "\t" << "Top Terms: ";
		for (int i = 0; i < numTerms; i++) {
			std::cout << terms[get<0>(topTerms[c][i])] << "(" << get<1>(topTerms[c][i]) << "), ";
		}
		std::cout << std::endl;
		std::cout << "\t" << "Top Docs: ";
		for (int i = 0; i < numDocs; i++) {
			std::cout << "\"" << docTitles[get<0>(topDocs[c][i])] << "\"(" << get<1>(topDocs[c][i]) << "), ";
		}
		std::cout << std::endl;
	}
}

void printTermRelated(const int termIdx, const int numTop, const std::vector<std::tuple<int, float>> termTerm, const std::vector<std::tuple<int, float>> termDoc, const vector<string> terms, const vector<string> docTitles) {
	auto count = 0;
	std::cout << "Top " << numTop << " terms correlates to term index " << termIdx << " :" << std::endl;
	std::cout << "\t";
	count = 0;
	for (auto & t : termTerm) {
		std::cout << terms[std::get<0>(t)] << "(" << std::get<1>(t) << "), ";
		count++;
		if (count >= numTop) {
			break;
		}
	}
	std::cout << std::endl;

	std::cout << "Top " << numTop << " docs correlates to term index " << termIdx << " :" << std::endl;
	std::cout << "\t";
	count = 0;
	for (auto & t : termDoc) {
		std::cout << "\"" << docTitles[std::get<0>(t)] << "\"(" << std::get<1>(t) << "), ";
		count++;
		if (count >= numTop) {
			break;
		}
	}
	std::cout << std::endl;
}

const int show_top = 5;

void lsa_demo(const int numDocs, const int numTerms, float * docTermFreq, const vector<string> docTitles, const vector<string> terms, const int numConcepts) {
	
	auto transposed = false;
	auto m = numDocs;
	auto n = numTerms;
	auto k = numConcepts;
	if (numDocs < numTerms) {
		transpose(numDocs, numTerms, docTermFreq);
		m = numTerms;
		n = numDocs;
		transposed = true;
	}
	tfidf(m, n, docTermFreq);
	auto u = unique_ptr<float>(new float[m * k]);
	auto s = unique_ptr<float>(new float[k]);
	auto vt = unique_ptr<float>(new float[k * n]);
	approximate_svd(m, n, k, docTermFreq, u.get(), s.get(), vt.get());
	if (transposed) {
		u.swap(vt);
		auto temp = m;
		m = n;
		n = temp;
	}
	auto v = unique_ptr<float>();
	v.swap(vt);
	transpose(k, n, v.get());
	// print concepts
	auto topTerms = topElementsInTopConcepts(n, k, v.get(), show_top, show_top);
	auto topDocs = topElementsInTopConcepts(m, k, u.get(), show_top, show_top);
	printConcepts(show_top, show_top, show_top, topDocs, topTerms, terms, docTitles);

	//corrolated
	multiplyByDiagonalMatrix(n, k, v.get(), s.get());
	rowsNormalized(n, k, v.get());
	multiplyByDiagonalMatrix(m, k, u.get(), s.get());
	rowsNormalized(m, k, u.get());
	while (true) {
		std::cout << std::endl;
		std::cout << "Query term: ";
		char buf[1024];
		std::cin.getline(buf, 1024);
		string term(buf);
		if (term == "!") {
			break;
		}
		auto termIdx = -1;
		for (auto i = 0; i < numTerms; i++) {
			if (terms[i] == term) {
				termIdx = i;
				break;
			}
		}
		if (termIdx < 0) {
			std::cout << "Term not exist." << std::endl;
			continue;
		}
		auto termTerm = topsForTerm(n, k, v.get(), n, k, v.get(), termIdx);
		auto termDoc = topsForTerm(m, k, u.get(), n, k, v.get(), termIdx);
		
		printTermRelated(termIdx, show_top, termTerm, termDoc, terms, docTitles);
	}

}

int main(int argc, char *argv[]){
	std::cout << "This is CUDA-DENSE-SMALL LSA demo for EE451 Team7 course project. CUDA-DENSE-LARGE & CUDA-SPARCE-LARGE is on the way. SPARK-DENSE-LARGE is ready else where. SPARK-SPARCE-LARGE is on the way." << std::endl;

	assert(argc == 2);
	int numDocs, numTerms;
	auto docTitles = vector<string>();
	auto terms = vector<string>();
	auto doc_term_freq_mat = read(argv[1], numDocs, numTerms, docTitles, terms);

	std::cout << "Read " << numDocs << " docs, " << numTerms << " terms." << std::endl;
	auto not_0_count = 0;
	for (auto i = 0; i < numDocs; i++) {
		for (auto j = 0; j < numTerms; j++) {
			if (doc_term_freq_mat.get()[i * numTerms + j] != 0) {
				not_0_count++;
			}
		}
	}
	std::cout << not_0_count / double(numDocs * numTerms) << " non zero element(s)." << std::endl;
	
	auto numConcepts = (numDocs < numTerms ? numDocs : numTerms) / 10;
	lsa_demo(numDocs, numTerms, doc_term_freq_mat.get(), docTitles, terms, numConcepts);
    return 0;
}
