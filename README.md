# Requirements (WIP)
(requires python3, pip, venv; instructions presume linux)
```
git clone https://github.com/rhofvendahl/visual_parse
cd visual_parse
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz
curl -o data/event2mind.tar.gz https://s3-us-west-2.amazonaws.com/allennlp/models/event2mind-2018.09.17.tar.gz
```
