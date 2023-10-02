
import mlrun
#import code.caseclus

@mlrun.handler()
def train_faq_caseclus(knowledge_base, stopwords, synonyms, compounds, model_name: str = 'faq_model'):    
    """
    A function which trains Caseclus model for FAQ
    """
    model_data={"name":"faq_model_caseclus"}
    mdl = code.caseclus.Caseclus(model_data, synonyms, knowledge_base, stopwords, compounds)
    mdl.fitModel()


