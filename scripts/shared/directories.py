import os

class dicrectories:
    knowledge = "embedding_files/knowledge_files/tm_imdb_20k"
    
    @staticmethod
    def pickle_by_id(path, id):
        return os.path.join(path , str(id) + '.pkl')

    
    @staticmethod
    def pickle_exist(path):
        return os.path.exists(path)



