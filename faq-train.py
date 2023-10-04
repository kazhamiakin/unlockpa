
import mlrun
sys.path.append("./code")
from caseclus import Caseclus

@mlrun.handler()
def train_faq_caseclus(knowledge_base, stopwords, synonyms, compounds, model_name: str = 'faq_model'):    
    """
    A function which trains Caseclus model for FAQ
    """
    model_data={"name":"faq_model_caseclus"}
    mdl = Caseclus(model_data, synonyms, knowledge_base, stopwords, compounds)
    mdl.fitModel()


