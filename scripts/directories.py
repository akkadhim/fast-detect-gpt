import os

class dicrectories:
    knowledge = "knowledge_files/IMDbKnowledge"
    
    @staticmethod
    def pickle_by_id(path, id):
        return os.path.join(path , str(id) + '.pkl')

    
    @staticmethod
    def pickle_exist(path):
        return os.path.exists(path)



