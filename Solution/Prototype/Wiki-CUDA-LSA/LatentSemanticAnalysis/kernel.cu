
#include<iostream>
#include<fstream>
#include<vector>
#include<tfidf.cuh>
#include<svd.cuh>
#include<string>
#include<cassert>
#include<algorithm>
#include<memory>
#include"lsa.cuh"

using std::vector;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::get;
using std::cout;
using std::endl;

//#define PRINT_MATRIX

void printMatrix(const char* name, int m, int n, const float*A) {
#ifdef PRINT_MATRIX
	std::cout << name << ":" << std::endl;
	for (int row = 0; row < m; row++) {
		for (int col = 0; col < n; col++) {
			float Areg = A[row * n + col];
			std::cout.width(15);
			std::cout << Areg;
		}
		std::cout << std::endl;
	}
#endif // PRINT_MATRIX
}

shared_ptr<float> read(const string filename, int & numDocs, int & numTerms, vector<string> & docTitles, vector<string> & terms) {
	std::ifstream f(filename);
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
	memset(term_freq_mat.get(), 0, sizeof(float) * numDocs * numTerms);
	for (auto docIdx = 0; docIdx < numDocs; docIdx++) {
		int numTermsInDoc;
		f >> numTermsInDoc;
		for (auto j = 0; j < numTermsInDoc; j++) {
			int termIdx, freq;
			f >> termIdx >> freq;
			term_freq_mat.get()[docIdx * numTerms + termIdx] = freq;
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

void lsa_demo(const int numDocs, const int numTerms, float * docTermFreq, const vector<string> docTitles, const vector<string> terms, const int numConcepts, const int numShowTop) {
		
	auto m = numDocs;
	auto n = numTerms;
	auto k = numConcepts;
	printMatrix("TF", m, n, docTermFreq);
	tfidf(m, n, docTermFreq);
	printMatrix("IDF", m, n, docTermFreq);
	auto transposed = false;
	if (numDocs < numTerms) {
		transpose(numDocs, numTerms, docTermFreq);
		m = numTerms;
		n = numDocs;
		transposed = true;
	}	
	auto u = unique_ptr<float>(new float[m * k]);
	auto s = unique_ptr<float>(new float[k]);
	auto vt = unique_ptr<float>(new float[k * n]);
	approximate_svd(m, n, k, docTermFreq, u.get(), s.get(), vt.get());
	auto v = unique_ptr<float>();
	v.swap(vt);
	transpose(k, n, v.get());
	if (transposed) {
		u.swap(v);
		auto temp = m;
		m = n;
		n = temp;
	}
	printMatrix("U", m, k, u.get());
	printMatrix("S", 1, k, s.get());
	printMatrix("V", n, k, v.get());
	// print concepts
	auto topTerms = topElementsInTopConcepts(n, k, v.get(), numShowTop, numShowTop);
	auto topDocs = topElementsInTopConcepts(m, k, u.get(), numShowTop, numShowTop);
	printConcepts(numShowTop, numShowTop, numShowTop, topDocs, topTerms, terms, docTitles);

	//corrolated
	multiplyByDiagonalMatrix(n, k, v.get(), s.get());
	printMatrix("VS", n, k, v.get());
	rowsNormalized(n, k, v.get());
	printMatrix("normVS", n, k, v.get());
	multiplyByDiagonalMatrix(m, k, u.get(), s.get());
	printMatrix("US", m, k, u.get());
	rowsNormalized(m, k, u.get());
	printMatrix("normUS", m, k, u.get());
	while (true) {
		try {
			std::cout << std::endl;
			std::cout << "Query term: ";
			char buf[1024];
			std::cin.getline(buf, 1024);
			string term(buf);
			//trim
			if (term.empty()) {
				break;
			}
			term.erase(0, term.find_first_not_of(" "));
			term.erase(term.find_last_not_of(" ") + 1);
			//lower case
			std::transform(term.begin(), term.end(), term.begin(), ::tolower);
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

			printTermRelated(termIdx, numShowTop, termTerm, termDoc, terms, docTitles);
		} catch (std::exception e) {
			std::cout << e.what() << std::endl;
		}
	}

}

int main(int argc, char *argv[]){
	std::cout << "This is CUDA-SPARCE-SMALL LSA demo for EE451 Team7 course project. CUDA-SPARCE-LARGE is on the way. SPARK-SPARCE-LARGE is ready else where." << std::endl;

	assert(argc >= 2);
	int numDocs, numTerms;
	auto docTitles = vector<string>();
	auto terms = vector<string>();
	auto filename = string(argv[1]);
	auto numConcepts = std::numeric_limits<int>::max();
	if (argc >= 3) {
		numConcepts = atoi(argv[2]);
	}
	auto numShowTop = 5;
	if (argc >= 4) {
		numShowTop = atoi(argv[3]);
	}
	cout << "filename=" << filename << "\t" << "#concept=" << numConcepts << "\t" << "#show=" << numShowTop << endl;
	
	auto doc_term_freq_mat = read(filename, numDocs, numTerms, docTitles, terms);
	std::cout << "Read " << numDocs << " docs, " << numTerms << " terms." << std::endl;
	auto minNumConcept = std::min(numDocs, numTerms);
	if (minNumConcept < numConcepts) {
		cout << "Note: #concept " << numConcepts << " -> " << minNumConcept << endl;
		numConcepts = minNumConcept;
	}
	auto minNumShowTop = std::min(numConcepts, numShowTop);
	if (minNumShowTop < numShowTop) {
		cout << "Note: #show " << numShowTop << " -> " << minNumShowTop << endl;
		numShowTop = minNumShowTop;
	}

	lsa_demo(numDocs, numTerms, doc_term_freq_mat.get(), docTitles, terms, numConcepts, numShowTop);
    return 0;
}
