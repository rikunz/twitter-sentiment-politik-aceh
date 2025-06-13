import re
import emoji
import urllib.request
import json
import streamlit as st
def load_secret(key):
    try:
        return st.secrets[key]
    except Exception:
        return None

character = ['.',',',';',':','-,','...','?','!','(',')','[',']','{','}','<','>','"','/','\'','#','-','@',
             'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def repeatcharClean(text):
    for i in range(len(character)):
        charac_long = 5
        while charac_long > 2:
            char = character[i]*charac_long
            text = text.replace(char, character[i])
            charac_long -= 1
    return text

def clean_review(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = emoji.demojize(text)
    text = re.sub(':[A-Za-z_-]+:', ' ', text)
    text = re.sub(r"([xX;:]'?[dDpPvVoO3)(])", ' ', text)
    text = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", "", text)
    text = re.sub(r"@[^\s]+[\s]?", ' ', text)
    text = re.sub(r"username", ' ', text)
    text = re.sub(r'#(\S+)', r'\1', text)
    text = re.sub('[^a-zA-Z,.?!]+',' ',text)
    text = repeatcharClean(text)
    text = re.sub('[ ]+',' ',text)
    return text

class Indobert:
    def __init__(self, api_url=None, api_key=None):
        self.api_url = load_secret("api_url") if api_url is None else api_url
        self.api_key = load_secret("api_key") if api_key is None else api_key

    def preprocess(self, text):
        return clean_review(text)

    def predict(self, text):
        text = self.preprocess(text)
        data = {"text": text}
        body = str.encode(json.dumps(data))
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Bearer ' + self.api_key
        }
        req = urllib.request.Request(self.api_url, body, headers)
        try:
            response = urllib.request.urlopen(req)
            result = response.read()
            result_json = json.loads(result)
            # Sesuaikan parsing ini dengan response dari endpoint Azure Anda
            label_predicted = result_json.get("prediction", None)
            score_predicted = result_json.get("scores").get(label_predicted, None)
            return {"label": label_predicted, "score": score_predicted}
        except urllib.error.HTTPError as error:
            print("The request failed with status code: " + str(error.code))
            print(error.info())
            return {"label": None, "score": None}