from newsapi import NewsApiClient
from PyDictionary import PyDictionary
import json
from config.global_vars import *

NEWS_API_KEY = 'ebccf0b6f44a4b95a1b8470811045581'
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

def get_news():
    try:
        inp = input(f"{txtcolor.LYDYA_SUCCESS} Please choose category (business,entertainment,technology): ")
        output = ''
        top_headlines = newsapi.get_top_headlines(category= inp,language='en',country='us',page_size=5)
        for articles in top_headlines['articles']:
            keys = ['title','description','url']
            for key in keys:
                output += key.upper()+': ' + articles[key] +'\n'
            output += '-'*30+'\n'
        return output
    except:
        return FUNC_FAILURE
    

def define_word():
    try:
        dictionary=PyDictionary()
        inp = input(f"{txtcolor.LYDYA_SUCCESS}Lydya: Please enter the word: ")
        return json.dumps(dictionary.meaning(inp),indent=2)
    except:
        return FUNC_FAILURE
    