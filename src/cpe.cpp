#include <iostream>
#include <vector>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

using namespace std;

// linked list node for holding a sequence of integers
struct node {
    int data = -1;
    node* next = NULL;
};

// in place merge of a linked list, and update the counter
void merge(node* head, pair<int, int> p, int new_token, map<pair<int, int>, int>& counter) {
    node* current = head;
    node* previous = NULL;
    while (current->next != NULL) {
        node* current_next = current->next;
        if (current->data == p.first && current_next->data == p.second) {
            // cout << "[DEBUG] merge chain: " << (previous != NULL ? previous->data : -1) << "--> [" << current->data << "-->" << current_next->data << "]-->" << (current_next->next != NULL ? current_next->next->data : -1) << endl;
            // inplace merge
            current->data = new_token;
            current->next = current_next->next; // maybe null
            // update counter
            if (current->next != NULL) {
                counter[{new_token, current->next->data}]++;
                counter[{p.second, current->next->data}]--;
            }
            if (previous != NULL) {
                counter[{previous->data, new_token}]++;
                counter[{previous->data, p.first}]--;
            }
            // delete the merged node
            delete current_next;
        }
        // move to the next
        previous = current;
        if (current->next != NULL) {
            current = current->next;
        } else {
            break; // end of the list
        }
    }
}

// we don't do any error checking here, so make sure the input is checked in python side!
class CPETokenizer {
public:
    // constructor
    CPETokenizer(int num_basic_tokens=128, int vocab_size=1024, bool verbose=true) {
        this->num_basic_tokens = num_basic_tokens;
        this->vocab_size = vocab_size;
        this->verbose = verbose;
    }

    int num_basic_tokens;
    int vocab_size;
    bool verbose;

    // the tokenizer model
    map<int, vector<int>> vocab;
    map<pair<int, int>, int> merges;

    // converted dataset in linked list format
    vector<node*> dataset;

    // counter of each pair
    map<pair<int, int>, int> counter;

    // import dataest
    void import_dataset(vector<vector<int>> corpus) {
        if (verbose) cout << "[INFO] Importing the dataset..." << endl;
        // convert corpus into linked list format
        for (int i = 0; i < corpus.size(); i++) {
            node* head = new node;
            head->data = corpus[i][0];
            node* current = head;
            for (int j = 1; j < corpus[i].size(); j++) {
                node* new_node = new node;
                new_node->data = corpus[i][j];
                current->next = new_node;
                current = new_node;
            }
            dataset.push_back(head);
        }
        if (verbose) cout << "[INFO] Dataset size: " << dataset.size() << endl;
    }

    // the dataset after training can be exported and cached for later use
    vector<vector<int>> export_dataset() {
        // convert the linked list dataset back to vector of vectors
        vector<vector<int>> result;
        for (int i = 0; i < dataset.size(); i++) {
            vector<int> sequence;
            node* current = dataset[i];
            while (current != NULL) {
                sequence.push_back(current->data);
                current = current->next;
            }
            result.push_back(sequence);
        }
        return result;
    }

    // clear the dataset
    void clear_dataset() {
        for (int i = 0; i < dataset.size(); i++) {
            node* current = dataset[i];
            while (current != NULL) {
                node* next = current->next;
                delete current;
                current = next;
            }
        }
        dataset.clear();
    }

    void train() {
        
        if (verbose) cout << "[INFO] Training the tokenizer..." << endl;

        // initialize the vocabulary with the basic tokens
        if (verbose) cout << "[INFO] Initializing the vocabulary with basic tokens..." << endl;
        vocab.clear();
        merges.clear();
        for (int i = 0; i < num_basic_tokens; i++) {
            vocab[i] = vector<int>{i};
        }

        // initialize the counter
        if (verbose) cout << "[INFO] Initializing the counter..." << endl;
        for (int i = 0; i < dataset.size(); i++) {
            node* current = dataset[i];
            while (current->next != NULL) {
                counter[{current->data, current->next->data}]++;
                current = current->next;
            }
        }

        // train the tokenizer
        int num_merges = vocab_size - num_basic_tokens;
        if (verbose) cout << "[INFO] target num_merges: " << num_merges << endl;

        for (int i = 0; i < num_merges; i++) {

            // find the most frequent pair (On)
            pair<int, int> p;
            int max_count = 0;
            for (auto it = counter.begin(); it != counter.end(); it++) {
                if (it->second > max_count) {
                    p = it->first;
                    max_count = it->second;
                }
            }

            // add a new token (O1)
            int new_token = num_basic_tokens + i;
            merges[p] = new_token;
            vocab[new_token] = vocab[p.first];
            vocab[new_token].insert(vocab[new_token].end(), vocab[p.second].begin(), vocab[p.second].end());

            if (verbose) cout << "[INFO] merging " << i << ": " << p.first << " + " << p.second << " -> " << new_token << " (count: " << max_count << ")" << endl;

            // merge in dataset and update counter (On)
            counter.erase(p);
            for (int j = 0; j < dataset.size(); j++) {
                merge(dataset[j], p, new_token, counter);
            }
        }
    }

    // On2, must merge from the lower index to the higher index
    vector<int> encode(vector<int> sequence) {
        // convert to linked list
        node* head = new node;
        head->data = sequence[0];
        node* current = head;
        for (int i = 1; i < sequence.size(); i++) {
            node* new_node = new node;
            new_node->data = sequence[i];
            current->next = new_node;
            current = new_node;
        }
        // count the pairs
        map<pair<int, int>, int> local_counter;
        current = head;
        while (current->next != NULL) {
            local_counter[{current->data, current->next->data}]++;
            current = current->next;
        }
        // merge
        while (true) {
            // find the merge pair with lowest index
            pair<int, int> p;
            int min_index = vocab_size;
            for (auto it = local_counter.begin(); it != local_counter.end(); it++) {
                if (merges.find(it->first) != merges.end() && merges[it->first] < min_index) {
                    p = it->first;
                    min_index = merges[p];
                }
            }
            // no more merges
            if (min_index == vocab_size) break;
            // merge
            local_counter.erase(p);
            merge(head, p, merges[p], local_counter);
        }
        // convert linked list back to vector
        vector<int> result;
        current = head;
        while (current != NULL) {
            result.push_back(current->data);
            current = current->next;
        }
        // delete linked list
        current = head;
        while (current != NULL) {
            node* next = current->next;
            delete current;
            current = next;
        }
        return result;
    }

    // On, just look up the vocab
    vector<int> decode(vector<int> sequence) {
        vector<int> result;
        for (int i = 0; i < sequence.size(); i++) {
            vector<int> token = vocab[sequence[i]];
            result.insert(result.end(), token.begin(), token.end());
        }
        return result;
    }

    // only merges are needed, vocab can be reconstructed from merges in On
    map<pair<int, int>, int> export_merges() {
        return merges;
    }

    // load merges
    void import_merges(vector<pair<int, int>> sorted_merges) {
        // reconstruct merges
        merges.clear();
        for (int i = 0; i < sorted_merges.size(); i++) {
            merges[sorted_merges[i]] = i + num_basic_tokens;
        }
        // reconstruct vocab
        vocab.clear();
        for (int i = 0; i < num_basic_tokens; i++) {
            vocab[i] = vector<int>{i};
        }
        for (int i = 0; i < sorted_merges.size(); i++) {
            vocab[i + num_basic_tokens] = vocab[sorted_merges[i].first];
            vocab[i + num_basic_tokens].insert(vocab[i + num_basic_tokens].end(), vocab[sorted_merges[i].second].begin(), vocab[sorted_merges[i].second].end());
        }
        // update vocab size
        vocab_size = num_basic_tokens + sorted_merges.size();
    }
};

// pybind code in the same cpp file, so we don't need to separate the header
PYBIND11_MODULE(_cpe, m) { // _cpe is the name of the compiled module (which is imported in python by: import _cpe)
    m.doc() = "Mesh Coordinate Pair Encoding (CPE) implementation in c++ with pybind11";

    py::class_<CPETokenizer>(m, "CPETokenizer")
        .def(py::init<int, int, bool>())
        .def_readwrite("num_basic_tokens", &CPETokenizer::num_basic_tokens)
        .def_readwrite("vocab_size", &CPETokenizer::vocab_size)
        .def("import_dataset", &CPETokenizer::import_dataset)
        .def("export_dataset", &CPETokenizer::export_dataset)
        .def("clear_dataset", &CPETokenizer::clear_dataset)
        .def("train", &CPETokenizer::train)
        .def("encode", &CPETokenizer::encode)
        .def("decode", &CPETokenizer::decode)
        .def("export_merges", &CPETokenizer::export_merges)
        .def("import_merges", &CPETokenizer::import_merges);
}