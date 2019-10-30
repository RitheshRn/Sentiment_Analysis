#!/bin/python

def combined(tarfname, unlabeled_size, LRclassifier, sentiment, unlabeled):

    import classify
    import numpy as np
    import scipy.sparse as sp
    for idx, sen in enumerate(unlabeled.data):
        unlabeled.data[idx] = unlabeled.data[idx].lower()
        
        
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    unlabeled_temp_Y = classify.evaluate_predict(unlabeled.X, LRclassifier)
    unlabeled_temp_Y_list = unlabeled_temp_Y.tolist()
    unlabeled_temp_Y_prob = np.amax(classify.evaluate_predict_prob(unlabeled.X, LRclassifier), axis=1).tolist()
    
    idk = {}
    for i in range(len(unlabeled.data)):
        idk[(unlabeled.data[i], unlabeled_temp_Y_list[i])] = unlabeled_temp_Y_prob[i]
    idk_sorted = sorted(idk.items(), key=lambda x: x[1], reverse=True)
    
    unlabeled.data[:] = [i[0][0] for i in idk_sorted]
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    unlabeled_temp_Y = classify.evaluate_predict(unlabeled.X, LRclassifier)
        
    # Add unsupervised data to Train
    sentiment.train_data += unlabeled.data[:unlabeled_size]
    unlabeled.data = unlabeled.data[unlabeled_size:]
    sentiment.trainX = sp.vstack((sentiment.trainX, unlabeled.X[:unlabeled_size]))
    sentiment.trainy = np.concatenate((sentiment.trainy, unlabeled_temp_Y[:unlabeled_size])) 
    unlabeled.X = unlabeled.X[unlabeled_size:]
        
    """
    CountVectorizer foloowed by TF-IDF
    """
    from nltk import word_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1, 3), tokenizer=word_tokenize)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)  
    

    return sentiment, unlabeled
    

def read_files(tarfname):
    """Read the training and development data from the sentiment tar file.
    The returned object contains various fields that store sentiment data, such as:

    train_data,dev_data: array of documents (array of words)
    train_fnames,dev_fnames: list of filenames of the doccuments (same length as data)
    train_labels,dev_labels: the true string label for each document (same length as data)

    The data is also preprocessed for use with scikit-learn, as:

    count_vec: CountVectorizer used to process the data (for reapplication on new data)
    trainX,devX: array of vectors representing Bags of Words, i.e. documents processed through the vectorizer
    le: LabelEncoder, i.e. a mapper from string labels to ints (stored for reapplication)
    target_labels: List of labels (same order as used in le)
    trainy,devy: array of int labels, one for each document
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    trainname = "train.tsv"
    devname = "dev.tsv"
    for member in tar.getmembers():
        if 'train.tsv' in member.name:
            trainname = member.name
        elif 'dev.tsv' in member.name:
            devname = member.name
            
            
    class Data: pass
    sentiment = Data()
    
    

    sentiment.train_data, sentiment.train_labels = read_tsv(tar,trainname)
    sentiment.dev_data, sentiment.dev_labels = read_tsv(tar, devname)


    """
    Lower case the sentences
    """
    for idx, sen in enumerate(sentiment.train_data):
        sentiment.train_data[idx] = sentiment.train_data[idx].lower()
    for idx, sen in enumerate(sentiment.dev_data):
        sentiment.dev_data[idx] = sentiment.dev_data[idx].lower()
    

    """
    Baseline
    """
#     from sklearn.feature_extraction.text import CountVectorizer
#     sentiment.count_vect = CountVectorizer()
#     sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
#     sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)

    """
    TfidfTransformer
    """
#     import numpy as np
#     from sklearn.feature_extraction.text import TfidfTransformer
#     sentiment.count_vect = TfidfTransformer()
#     sentiment.train_data = np.expand_dims(np.array(sentiment.train_data), axis=1)
#     sentiment.dev_data = np.expand_dims(np.array(sentiment.dev_data), axis=1)
#     sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
#     sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    
    """
    Lemmatization
    
    
    
    import nltk
    #nltk.download('averaged_perceptron_tagger')
    import nltk
    #nltk.download('wordnet')
    import nltk
    #nltk.download('punkt')
    from nltk.corpus import wordnet
    
    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None
    
    def lemmatize_sentence(sentence):
        #tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:        
                #else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)
    
    # Train data lemmetization
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    sentiment.train_data[:] = [lemmatize_sentence(i) for i in sentiment.train_data]
    
    # Dev data lemmetization
    sentiment.dev_data[:] = [lemmatize_sentence(i) for i in sentiment.dev_data]
    """
    
#     # Stemming
    
#     from nltk.stem import PorterStemmer
#     porter_stemmer=PorterStemmer()
#     sentiment.train_data[:] = [porter_stemmer.stem(word=i) for i in sentiment.train_data]
#     sentiment.dev_data[:] = [porter_stemmer.stem(word=i) for i in sentiment.dev_data]
    
    
    """
    CountVectorizer foloowed by TF-IDF
    """
    from nltk import word_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    sentiment.count_vect = TfidfVectorizer(ngram_range=(1, 3), tokenizer=word_tokenize)
    sentiment.trainX = sentiment.count_vect.fit_transform(sentiment.train_data)
    sentiment.devX = sentiment.count_vect.transform(sentiment.dev_data)
    
    
    """
    Word2vec
    """
    
    
#     ### Tokenizer
#     from nltk.tokenize import TweetTokenizer
#     tknzr = TweetTokenizer()
    
#     # Tokennize train data
#     sentiment.train_data[:] = [tknzr.tokenize(i) for i in sentiment.train_data]
#     # Tokennize dev data
#     sentiment.dev_data[:] = [tknzr.tokenize(i) for i in sentiment.dev_data]
    
    
    
#     from gensim.test.utils import common_texts, get_tmpfile
#     from gensim.models import Word2Vec
    
    
    
    
# #     try:
# #         from gensim.models import KeyedVectors
# #         path = get_tmpfile("wordvectors.kv")
# #         model.wv.save(path)
# #         wv = KeyedVectors.load("model.wv", mmap='r')
# #     except:
        
#     print("started word2vec")
    
    
    
#     try:
#         model = Word2Vec.load("word2vec.model")
#     except:
        
#         model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
#         #model.train(sentiment.train_data, total_examples=4582, epochs=1)
#         model.save("word2vec.model")
        
    
    
        
#     def get_word2vec(data): 
#         holder = []
#         for s_idx, sentence in enumerate(data):
#             min_vector = []
#             max_vector = []
#             word_vector_matrix = []
#             for word in sentence:
#                 try:
#                     word_vector_matrix.append(model.wv[word.lower()])
#     #                 print(word_vector_matrix)
#     #                 import sys
#     #                 sys.exit()
#                 except:
#                     word_vector_matrix.append([0 for i in range(100)])
#             word_vector_matrix_t = [[word_vector_matrix[j][i] for j in range(len(word_vector_matrix))] for i in range(len(word_vector_matrix[0]))]
             
#             try: 
#                 for j in range(100):
#                     min_vector.append(min(word_vector_matrix_t[j]))
#                     max_vector.append(max(word_vector_matrix_t[j]))
#             except:
#                 print(sentence)
#                 print(np.array(word_vector_matrix).shape)
#                 print(np.array(word_vector_matrix_t).shape)
#                 print()
#                 import sys
#                 sys.exit()


#             holder.append( min_vector + max_vector)
#         return holder
    
#     sentiment.trainX = get_word2vec(sentiment.train_data)
#     sentiment.devX = get_word2vec(sentiment.dev_data)
        
        
        
#     print("end of word2vec")
    
    
    
    from sklearn import preprocessing
    sentiment.le = preprocessing.LabelEncoder()
    sentiment.le.fit(sentiment.train_labels)
    sentiment.target_labels = sentiment.le.classes_
    sentiment.trainy = sentiment.le.transform(sentiment.train_labels)
    sentiment.devy = sentiment.le.transform(sentiment.dev_labels)
    

    
    tar.close()
    return sentiment


def read_unlabeled(tarfname, sentiment):
    """Reads the unlabeled data.

    The returned object contains three fields that represent the unlabeled data.

    data: documents, represented as sequence of words
    fnames: list of filenames, one for each document
    X: bag of word vector for each document, using the sentiment.vectorizer
    """
    import tarfile
    tar = tarfile.open(tarfname, "r:gz")
    class Data: pass
    unlabeled = Data()
    unlabeled.data = []
    
    unlabeledname = "unlabeled.tsv"
    for member in tar.getmembers():
        if 'unlabeled.tsv' in member.name:
            unlabeledname = member.name
            
#     print(unlabeledname)
    tf = tar.extractfile(unlabeledname)
    for line in tf:
        line = line.decode("utf-8")
        text = line.strip()
        unlabeled.data.append(text)
        
    """
    Lower case the sentences
    """
    for idx, sen in enumerate(unlabeled.data):
        unlabeled.data[idx] = unlabeled.data[idx].lower()
        
        
        
    """
    Lemmatization
    
    
    
    import nltk
    #nltk.download('averaged_perceptron_tagger')
    import nltk
    #nltk.download('wordnet')
    import nltk
    #nltk.download('punkt')
    from nltk.corpus import wordnet
    
    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None
    
    
    
    def lemmatize_sentence(sentence):
        #tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
        #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
                #if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:        
                #else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)
    
    # Train data lemmetization
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    unlabeled.data[:] = [lemmatize_sentence(i) for i in unlabeled.data]
    """
    
#     # Stemming
    
#     from nltk.stem import PorterStemmer
#     porter_stemmer=PorterStemmer()
#     unlabeled.data[:] = [porter_stemmer.stem(word=i) for i in unlabeled.data]
    
    
    
#     # Word2Vec
#     from gensim.test.utils import common_texts, get_tmpfile
#     from gensim.models import Word2Vec
    
#     model = Word2Vec.load("word2vec.model")
    
#     for s_idx, sentence in enumerate(unlabeled.data):
#         min_vector = []
#         max_vector = []
#         word_vector_matrix = []
#         for word in sentence:
            
#             try:
#                 word_vector_matrix.append(model.wv[word.lower()])
#             except:
#                 word_vector_matrix.append([0 for i in range(100)])          
#         word_vector_matrix_t = [[word_vector_matrix[j][i] for j in range(len(word_vector_matrix))] for i in range(len(word_vector_matrix[0]))]
#         for j in range(100):
#             min_vector.append(min(word_vector_matrix_t[j]))
#             max_vector.append(max(word_vector_matrix_t[j]))
        
#         unlabeled.data[s_idx] = min_vector + max_vector
    
#     unlabeled.X = unlabeled.data[:]
    
    
    
    unlabeled.X = sentiment.count_vect.transform(unlabeled.data)
    
    
    
    
#     print(unlabeled.X.shape)
    tar.close()
    return unlabeled

def read_tsv(tar, fname):
    member = tar.getmember(fname)
#     print(member.name)
    tf = tar.extractfile(member)
    data = []
    labels = []
    for line in tf:
        line = line.decode("utf-8")
        (label,text) = line.strip().split("\t")
        labels.append(label)
        data.append(text)
    return data, labels

def write_pred_kaggle_file(unlabeled, cls, outfname, sentiment):
    """Writes the predictions in Kaggle format.

    Given the unlabeled object, classifier, outputfilename, and the sentiment object,
    this function write sthe predictions of the classifier on the unlabeled data and
    writes it to the outputfilename. The sentiment object is required to ensure
    consistent label names.
    """
    yp = cls.predict(unlabeled.X)
    labels = sentiment.le.inverse_transform(yp)
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    for i in range(len(unlabeled.data)):
        f.write(str(i+1))
        f.write(",")
        f.write(labels[i])
        f.write("\n")
    f.close()


def write_gold_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the truth.

    You will not be able to run this code, since the tsvfile is not
    accessible to you (it is the test labels).
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write(label)
            f.write("\n")
    f.close()

def write_basic_kaggle_file(tsvfile, outfname):
    """Writes the output Kaggle file of the naive baseline.

    This baseline predicts POSITIVE for all the instances.
    """
    f = open(outfname, 'w')
    f.write("ID,LABEL\n")
    i = 0
    with open(tsvfile, 'r') as tf:
        for line in tf:
            (label,review) = line.strip().split("\t")
            i += 1
            f.write(str(i))
            f.write(",")
            f.write("POSITIVE")
            f.write("\n")
    f.close()

if __name__ == "__main__":
    regularization_constant_dict_train = {}
    regularization_constant_dict_dev = {}
    
    import classify
    best_val = 0
    threshold = 9000
    tarfname = "data/sentiment.tar.gz"
    sentiment = read_files(tarfname)
    unlabeled = read_unlabeled(tarfname, sentiment)
    cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 100)
    
    for i in range(2):

        regularization_constant_dict_train[i] = classify.evaluate(sentiment.trainX, sentiment.trainy, cls, 'train')
        regularization_constant_dict_dev[i] = classify.evaluate(sentiment.devX, sentiment.devy, cls, 'dev')
        if i ==1:
            import numpy as np

            y_pred = classify.evaluate_predict(sentiment.devX, cls) #, axis=1).tolist()
            import pandas as pd
            df = pd.DataFrame({'sent':sentiment.dev_data, 'y_true':sentiment.devy, 'y_pred':y_pred})
            df.to_csv('output.csv')
        
        sentiment, unlabeled = combined(tarfname, threshold, cls, sentiment, unlabeled)
        
        cls = classify.train_classifier(sentiment.trainX, sentiment.trainy, 1000)

        tfidf_feature_list = sentiment.count_vect.get_feature_names()
        cls_weights = cls.coef_.tolist()

        """
        Uncomment to get the most weighted Positive and Negative words
        """
#         d = {tfidf_feature_list[i]: cls_weights[0][i] for i in range(len(tfidf_feature_list))}
#         print("\nMost weighted : ")
#         print(sorted(d.items(), key=lambda x: abs(x[1]), reverse = True)[:5])
#         print("\nPositive : ")
#         print(sorted(d.items(), key=lambda x: x[1], reverse = True)[:5])
#         print("\nNegative : ")
#         print(sorted(d.items(), key=lambda x: x[1])[:5])
        

        
# #         if regularization_constant_dict_dev[i] > best_val:
# #             best_val = regularization_constant_dict_dev[i]
# #         if best_val == regularization_constant_dict_dev[i]:
# #         print("\nReading unlabeled data")
#         unlabeled = read_unlabeled(tarfname, sentiment)
# #         print("Writing predictions to a file")
#         write_pred_kaggle_file(unlabeled, cls, "data/sentiment-pred_"+str(i)+".csv", sentiment)
#     #write_basic_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-basic.csv")
    
    
#     print(sorted(regularization_constant_dict_train.items(), key = lambda x: x[1], reverse=True))
#     print(sorted(regularization_constant_dict_dev.items(), key = lambda x: x[1], reverse=True))
#     print(regularization_constant_dict_dev)
    # You can't run this since you do not have the true labels
    # print "Writing gold file"
    # write_gold_kaggle_file("data/sentiment-unlabeled.tsv", "data/sentiment-gold.csv")
