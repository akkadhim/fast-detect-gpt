from setuptools import setup, find_packages
from setuptools.command.install import install
import nltk
from nltk.data import find

class CustomInstallCommand(install):
    def run(self):
        # Perform the normal installation process
        install.run(self)
        try:
            find('corpora/wordnet.zip')
        except LookupError:
            print("Downloading NLTK 'wordnet' data...")
            nltk.download('wordnet')

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()
    
setup(
   name='Embedding-Adversarial-Attack',
   version='0.1.0',
   author='Ahmed Khalid',
   author_email='ahmed.k.kadhim@uia.no',
   url='',
   license='MIT',
   description='Embedding Adversarial Attack',
   long_description='Adversarial Attack on AI-Text Detection system using Embedding Models',
   keywords ='pattern-recognition cuda machine-learning interpretable-machine-learning rule-based-machine-learning propositional-logic tsetlin-machine cybersecurity adversarial-attacks gpt detect-gpt',
   packages=find_packages(),  
   install_requires=parse_requirements('requirements.txt'),
   cmdclass={
        'install': CustomInstallCommand,
    },
)

