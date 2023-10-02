class ServiceError(Exception):

    def __init__(self, code, description):
        self.code = code
        self.description = description

    @property
    def dict(self):
        return {
            'code': self.code,
            'description': self.description
        }

# ==================== Train Service Errors (40) ====================


class TrainGenericError(ServiceError):

    def __init__(self, description=None):
        self.code = 40
        self.description = description or 'Unknown error in the service train'


class PathNotFoundError(ServiceError):

    def __init__(self, path):
        self.code = 41
        self.description = f'Directory {path} not found'


class ModelNotFoundError(ServiceError):

    def __init__(self, model_id):
        self.code = 42
        self.description = f'Model {model_id} not found'


class ModelNotTrained(ServiceError):

    def __init__(self, model_id):
        self.code = 43
        self.description = f'Model {model_id} is created but not trained yet'


class ModelTopicsError(ServiceError):
    def __init__(self, model_id, ntopics, qnumb):
        self.code = 44
        self.description = f'Model {model_id} not trained: topic numbers {ntopics} must be less than number of questions {qnumb}'


class ModelClusterError(ServiceError):
    def __init__(self, model_id, ncluster, qnumb):
        self.code = 45
        self.description = f'Model {model_id} not trained: cluster numbers {ncluster} must be less than number of questions {qnumb}'


class W2vFileError(ServiceError):
        def __init__(self, model_id, file):
            self.code = 46
            self.description = f'Word2vec input file {file} not found or not well-formatted for Model {model_id}'

# ==================== Insert Service Errors (50) ====================