from base import Base
from exceptions import ModelNotTrained, ModelTopicsError, ModelClusterError
from numpy.random import seed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from utils import *
import nltk

from .spacyutil import spacyHelper, vectorize_util
from mlrun.frameworks.sklearn import apply_mlrun

class Caseclus(Base):

    def __init__(self, model_data, synonyms, knowledgeBase, stopwords, compounds):
        ## Chiamo il costruttore base per inizializzare le variabili comuni:
        super().__init__(model_data, synonyms, knowledgeBase, stopwords, compounds)

        ## Inizializzo le variabili specifiche della classe, utilizzando dove possibile gli
        ## oggetti già inizializzati:
        self.n_topics = self.model_data['n_topics']
        self.n_clusters =  self.model_data['n_cluster']

        # is_cluster and is_lsa_sim parameters can be used to select different approaches:
        #   - is_cluster = false and is_lsa_sim = false -> Approach I: similarity using TF-IDF matrix
        #   - is_cluster = false and is_lsa_sim = TRUE -> Approach IIa: similarity using topic of LSA matrix
        #   - is_cluster = TRUE and is_lsa_sim = false -> Approach IIb: clustering and similarity using TF-IDF
        #   - is_cluster = TRUE and is_lsa_sim = TRUE -> Approach IIc: clustering and similarity using topic of LSA matrix
        self.is_cluster = self.model_data['is_cluster']
        self.is_lsa_sim = self.model_data['is_lsa_sim']

        self.k = self.model_data['confidence_lsa'] if self.is_lsa_sim else self.model_data['confidence_tfidf']
        self.clusterModel = None
        self.spacy_nlp=spacyHelper.spacyInitializer(self.stopwordsList)


    def fitModel(self):
        """
        fitModel learn the internal model from the input data
        """
        seed(50)


        # Replace special characters
        self.df['DOMANDA_CLEAN'] = self.df['question'].apply(lambda x: str.replace(x, '?', ' ')).apply(lambda x: str.replace(x, '/', ' ')).apply(lambda x: str.replace(x, "’", ' ')).apply(lambda x: str.replace(x, "é", 'è')).apply(lambda x: str.replace(x, "€", 'euromoneta'))
        self.df['RISPOSTA_CLEAN'] = self.df['answer'].apply(lambda x: str.replace(x, '?', ' ')).apply(lambda x: str.replace(x, '/', ' ')).apply(lambda x: str.replace(x, "’", ' ')).apply(lambda x: str.replace(x, "é", 'è')).apply(lambda x: str.replace(x, "€", 'euromoneta'))

        # Create document corpus consisting of questions
      
        documents= self.df['DOMANDA_CLEAN']
        documents= documents.str.lower()
        # Create document corpus consisting of both answers and questions
        
        complete_document = self.df['DOMANDA_CLEAN'] + ' ' + self.df['RISPOSTA_CLEAN']
        complete_document= complete_document.str.lower()

        ## Parameter definition
        # Lemmatizer, stemmer, punctuation handling, compound words, synonyms
        self.stemmer = nltk.stem.snowball.ItalianStemmer()
        self.parComposte = dict(self.dfComp.to_dict('split')['data'])
        self.sinonimi = dict(self.dfSyn.to_dict('split')['data'])

        # Learn model:
        #   Step 1: vectorizer
        #   Step 2: tfidf
        #   Step 3: SVD decomposition
        #   Dictionary

        #substitute compounds words : new version
        
        documents=vectorize_util.SubstituteCompoundWords_vectorized(documents, self.parComposte)
        documents=documents.fillna("")

        _spacylemmacountvectorizer = spacyHelper.SpacyLemmaCountVectorizer(self.spacy_nlp,self.sinonimi,self.stemmer, ngram_range=(1, 2))#, max_df=0.98, min_df=2)
   
        tfidfTran = TfidfTransformer()

        if self.model_data['n_topics'] <= 1 or self.model_data['n_topics'] >= self.df.shape[0]:
            self.tagger = None
            raise ModelTopicsError(self.model_data['id'], self.model_data['n_topics'], self.df.shape[0])
        else:
            nTopics = self.n_topics

        lsa_model = TruncatedSVD(n_components = nTopics)

        # Select which model to use as similarity model
        # is_lsa_sim true -> Approach II: similarity evaluated using LSA matrix
        if self.is_lsa_sim:
            model = Pipeline(steps=[('vectorizer', _spacylemmacountvectorizer),
                    ('tfidf', tfidfTran),
                    ('lsa',lsa_model)])
        else: # is_lsa_sim false -> Similarity evaluated using TF-IDF matrix
            model = Pipeline(steps=[('vectorizer', _spacylemmacountvectorizer),
                    ('tfidf', tfidfTran)])

        # -------------------- The only line you need to add for MLOps -------------------------
        # Wraps the model with MLOps (test set is provided for analysis & accuracy measurements)
        apply_mlrun(model=model, model_name=self.model_data.model_name)
        # --------------------------------------------------------------------------------------
            
        # Fit models and retrieve FAQ matrix representation
        self.model = model.fit(documents)
        self.matrix = self.model.transform(documents)

        ## Get clusters
        if self.model_data['is_cluster']:
            if self.model_data['n_cluster'] <= 1 or self.botData['n_cluster'] >=  self.df.shape[0]:
                self.tagger = None
                raise ModelClusterError(self.model_data['id'], self.model_data['n_cluster'], self.df.shape[0])
            else:
                nCluster = self.n_clusters

            kmean_model = KMeans(n_clusters=nCluster, random_state=0)

            clusModel = Pipeline(steps=[('vectorizer', _spacylemmacountvectorizer),
                    ('tfidf', tfidfTran),
                    ('lsa',lsa_model),
                    ('kmean', kmean_model)])

            # -------------------- The only line you need to add for MLOps -------------------------
            # Wraps the model with MLOps (test set is provided for analysis & accuracy measurements)
            apply_mlrun(model=clusModel, model_name=self.model_data.model_name)
            # --------------------------------------------------------------------------------------

            self.clusterModel = clusModel.fit(documents)

            self.df['cluster'] = clusModel.named_steps['kmean'].labels_


    def predictOutput(self, queryInput):
        """
        Predict answers with similarity > k
        :param queryInput: text of query (str)
        :return: answers (list of dicts)
        """

        if self.model is None:
            raise ModelNotTrained(self.model_data['id'])


        queryInput = queryInput.lower()
        queryInputCompound=SubstituteCompoundWords(queryInput,self.parComposte)


        queryInput = [queryInputCompound.replace("€", "euromoneta").replace("$", "dollaro").replace("?","")]

        # Apply model
        newQuery = self.model.transform(queryInput)
        #print('NEWQUERY TYPE CLUS',type(newQuery))

        # Evaluate cosine similarity
        cos_TfIdfsim = cosine_similarity(newQuery, self.matrix)
        #print('COSCLUS', cos_TfIdfsim)
        #print('COSCLUS TYPE', type(cos_TfIdfsim))
        currentAnswer = addSimilarity(self.df, cos_TfIdfsim)
        selectedAnswers = currentAnswer[(currentAnswer.similarity >= self.k)]

        # is_cluster discriminates if kmeans clustering must be used
        # if is_cluster equal false -> Only similarity is evaluated
        # if is_cluster equal true -> Clustering used as filter
        if self.is_cluster:
            clusterQuery = self.clusterModel.predict(queryInput)[0]
            print('Domanda utente appartiene al cluster {}'.format(clusterQuery))
            # Return list of answers with similarity >= k and belonging to the same aestimated cluster
            selectedAnswers = selectedAnswers[selectedAnswers.cluster == clusterQuery ]

        return [
            {
                'text': selectedAnswers.answer.values[i],
                'confidence': selectedAnswers.similarity.values[i],
                'index_ques':selectedAnswers.question_number.values[i]
            } for i in range(len(selectedAnswers.similarity.values))
        ]
    def vectorOutput(self, queryInput):
        """
        Predict answers with similarity > k
        :param queryInput: text of query (str)
        :return: answers (list of dicts)
        """
        if self.model is None:
            raise ModelNotTrained(self.model_data['id'])

        queryInputCompound=SubstituteCompoundWords(queryInput,self.parComposte)
        queryInput = [queryInputCompound.replace("€", "euromoneta").replace("$", "dollaro")]

        # Apply model
        newQuery = self.model.transform(queryInput)
        return newQuery
