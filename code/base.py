#from nltk.corpus import stopwords
from spacy.lang.it.stop_words import STOP_WORDS
import string
class Base:


    def verifyDataframes(self, model_data, synonyms, knowledge_base, stopwordsInput, compounds):
        """
        Verify data frame coherence
        :param model_data: mandatory 
        :param synonyms: could be empty (pd.dataframe)
        :param knowledge_base: mandatory (pd.dataframe)
        :param stopwordsInput: could be empty (pd.dataframe)
        :param compounds: could be empty (pd.dataframe)
        """
        # Check data frames:
        # model and knowledgeBase data frames must not be empty
        if knowledge_base.empty:
            raise Exception('Empty data frame KNOWLEDGE BASE. Procedure stopped.')
        if model_data is None:
            raise Exception('Empty model data. Procedure stopped.')

        # Warning if other data frames Empty:
        if compounds.empty:
            print('Data frame COMPOUNDS empty for model ' + str(model_data['model_name']) + '.')
        if synonyms.empty:
            print('Data frame SYNONYMS empty for  model ' + str(model_data['model_name']) + '.')
        if stopwordsInput.empty:
            print('Data frame STOPWORDS empty for model ' + str(model_data['model_name']) + '.')

    def set_n_answers(self, n):
        """
        Set maximum number of similar answer to retrieve from query method
        :param n: number of anwsers to retrieve (int)
        :return:
        """
        self.n = n

    def __init__(self, model_data, synonyms, knowledgeBase, stopwordsInput, compounds):
        """
        Initialize class from data frames
        :param model_data: mandatory 
        :param sysnonyms: synonims (pd.dataframe)
        :param knowledgeBase: q&a (pd.dataframe)
        :param stopwordsInput: stopwords (pd.dataframe)
        :param compounds: compound words (pd.dataframe)
        """
        self.verifyDataframes(model_data, synonyms, knowledgeBase, stopwordsInput, compounds)

        # Keep only necessary columns
        self.df = knowledgeBase
        self.dfComp = compounds[['compound_word', 'base_token']]
        self.dfSyn = synonyms[['synonym_word', 'base_token']]
        self.model_data = model_data
        self.n = 5
        self.remove_punct_dict = dict((ord(punct), ' ') for punct in string.punctuation)

        # Set stopwords:
        wordsToAddDF = stopwordsInput[(stopwordsInput['to_keep'] == False)]
        wordsToRemoveDF = stopwordsInput[(stopwordsInput['to_keep'] == True)]
        wordsToAdd = []
        wordsToRemove = []
        for word in wordsToAddDF['stopword']:
            wordsToAdd.append(word)
        for word in wordsToRemoveDF['stopword']:
            wordsToRemove.append(word)
        self.stopwordsList = set(list(STOP_WORDS)+ [''] + wordsToAdd).difference(wordsToRemove)

        # Model attributes
        self.dizionario = None
        self.model = None
        self.matrix = None

    def fitModel(self):
        """
        Model fitting from input dataset
        """
        pass

    def predictOutput(self, queryInput):
        """
        Predict answers with similarity > k
        Override this method from every model
        :param queryInput: text of query (str)
        :return: answers (list of dicts)
        """
        return []

    def query(self, text, tenant='-1'):
        """
        Get input query and return list of max n answer
        :param text: text of query (str)
        :return: answers (list of dicts)
        """
        null_answer={
                        'text': str(-1),
                        'confidence': 0
                    }
        if text is not None and len(text)>=4:
            result = self.predictOutput(text)[:self.n]
            if len(result) == 0:
                result.append(null_answer)
        else:
            result=[]
            result.append(null_answer)
        
         
        return result
