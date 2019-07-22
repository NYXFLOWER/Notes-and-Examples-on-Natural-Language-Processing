import numpy as np


class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self, index, termWeighting):
        """
        :param index: dictionary - {key[word] : dictionary{key[index]: [count]}}
        :param termWeighting: 'binary', 'tf', 'tfidf'
        """
        self.index = index
        self.n_d = self.computeNumOfDocument()
        self.words = list(index.keys())     # index the words
        self.termWeighting = termWeighting
        self.coordinates, self.idfs = self.initTerms()  # normalized coordinates and inverse document frequency: array

    def computeNumOfDocument(self):
        documents = set()
        for value in self.index.values():
            for key in value.keys():
                documents.add(key)
        return len(documents)

    def initTerms(self):
        """
        For each object of this class, this method run once, to create the two-dimension array of the file 'index'
        :return: 1. the normalized term coordinates based on 'self.terWeighting'
        :return: 2. the idf of whole words in the collection
        """
        n_word = len(self.words)
        idf = 0
        coordinates = np.zeros(shape=(self.n_d, n_word), dtype=float)      # 3204 documents in total
        for i in range(n_word):
            doc_count = self.index[self.words[i]]          # {document_index: count}
            for doc in doc_count.keys():
                if self.termWeighting == 'binary':
                    coordinates[doc - 1, i] = 1
                else:
                    coordinates[doc - 1, i] = doc_count[doc]      # for the case 'tf' and 'tfidf'
        if self.termWeighting == 'tfidf':
            df = np.count_nonzero(coordinates, axis=0)    # array(n_word)
            idf = np.log10(n_word/df)     # array(n_word)
            coordinates *= idf          # (n_docu, n_word)
        # normalise the coordinate
        norm = np.sqrt(np.sum(coordinates**2, axis=1)).reshape((self.n_d, 1))     #(n_doc, 1)
        return coordinates/norm, idf    #.reshape((n_word, 1))

    def initQuery(self, query):
        """
        This function is used to normalized the coordinate of the input query
        :param query: type - dict {words: counts}
        :return: 1. the index of the words occur in the query
        :return: 2. the normalized query coordinate with 'self.termWeighting'
        """
        word_query = list(query.keys() & self.index.keys())
        n_word = len(word_query)
        index_word = np.zeros((1, n_word), dtype=int)
        coordinate_query = np.zeros((n_word, 1))

        # normalise for different weightings
        for i in range(n_word):
            index_word[0, i] = self.words.index(word_query[i])
            if self.termWeighting == 'binary':
                coordinate_query[i, 0] = 1
            else:
                coordinate_query[i, 0] = query[word_query[i]]
        if self.termWeighting == 'tfidf':
            idf_query = self.idfs[index_word.tolist()]
            coordinate_query *= np.array(idf_query).reshape((n_word, 1))
        return index_word, coordinate_query

    # Method performing retrieval for specified query
    def forQuery(self, query):
        """
        :param query: the representation of individual queries. {key[query-terms] : [counts]}
        :return: type-[], the index of document which is top-ten related with the input query
        """
        words_query, coordinate_query = self.initQuery(query)
        coordinate_selected = self.coordinates[:, words_query]      # (n_doc, n_word_query)
        similarity = np.dot(coordinate_selected, coordinate_query).flatten()
        return np.argpartition(similarity, -10)[-10:]+1


