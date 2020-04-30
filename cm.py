import numpy as np
import pandas as pd
import nltk
#nltk.download('punkt') # one time execution
import re
#f1=open('/home/diptanshu/Desktop/google.txt',mode='r')
#file=open('/home/diptanshu/Documents/checkmate/chetti.txt',mode='r')
#a1=f1.read()
#all_of_it=file.read()
#print(all_of_it)
#file.close()
def abc(all_of_it):                                                                         # input is string containing all the data, needs to be read from pdf
    import numpy as np
    import pandas as pd
    import nltk
    #nltk.download('punkt') # one time execution
    import re
    #f1=open('/home/diptanshu/Desktop/google.txt',mode='r')
    #file=open('/home/diptanshu/Documents/checkmate/chetti.txt',mode='r')
    #a1=f1.read()
    #all_of_it=file.read()
    #print(all_of_it)
    #file.close()
    from nltk.tokenize import sent_tokenize
    sentences=[]    
    #sentences.append(sent_tokenize(a1))
    sentences.append(sent_tokenize(all_of_it))
    sentences = [y for x in sentences for y in x]
    #print(sentences)
    word_embeddings = {}
    f = open('/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    #print(len(word_embeddings))
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    word_embeddings = {}
    f = open('love.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    sentence_vectors = []
    #print(len(clean_sentences))
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    sim_mat = np.zeros([len(sentences), len(sentences)])
    from sklearn.metrics.pairwise import cosine_similarity
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
    import networkx as nx

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    c=0
    a=""
    for i in ranked_sentences:
        if "collect" in (i[1]) or "data" in (i[1]) or "use" in (i[1])  or "access" in (i[1]) or "time" in i[1] and "party" in (i[1]):
            a+=i[1]+'\n'
            c+=1
        if c==9:
            break
    return a
