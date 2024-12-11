FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Install Python and other dependencies
RUN apt-get update && apt-get install -y python3-pip git

COPY . /app
WORKDIR /app


# You should have a requirements.txt to define your dependencies
# RUN pip3 install .

CMD [ "tail", "-f", "/dev/null" ]

# apt update
# apt-get install sudo
# https://www.geeksforgeeks.org/download-and-install-python-3-latest-version/
# update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 
# apt-get update
# apt-get install -y python3.11-dev
# apt-get install -y g++ gcc python3-dev
# numpy torch tqdm datasets transformers nltk fasttext gensim tensorflow tensorflow_hub