"""
What? Test preprocessing utilities

Utilies include:
    [1] relevant_pages
    [2] text_extractor
    [3] text_preprocessing_pipeline_1
    [4] text_preprocessing_pipeline_2
    [5] remove_custom_stopwords_unigrams
    [6] create_bigram
    [7] remove_custom_stopwords_bigrams
    [8] qa_text_processing_pipeline
    
Reference: https://mikechoi90.medium.com/investigating-microsofts-transformation-under-satya-nadella-f49083294c35
"""

# Import
import PyPDF2 as pdf
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter
import io
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict
from nltk import pos_tag
from nltk.util import ngrams
from spellchecker import SpellChecker


def relevant_pages(file_name):
    '''
    input: a PDF file 
    output: a PDF file excluding the first three pages
    '''
    file = open(file_name, 'rb')
    pdf_reader = pdf.PdfFileReader(file)
    pdf_writer = pdf.PdfFileWriter()
    for i in range(3,pdf_reader.getNumPages()-1):
        page_i = pdf_reader.getPage(i)
        pdf_writer.addPage(page_i)
    
    output = open('../Data/Transcripts/Pages.pdf','wb')
    pdf_writer.write(output)
    output.close()
    
def text_extractor(file_name):
    '''
    input: a file name of an earnings transcript
    output: extracted text from the transcript
    '''  
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    
    with open(file_name, 'rb') as fh:

        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
            
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    
    return text

def text_preprocessing_pipeline_1(dict):
    '''
    Input: a dictionary with names of docs as keys and transcript text as values
    Output: a dictionary with preprocessed transcript text (phase 1)
    ''' 
    ## Remove line breaks, punctuations and numbers
    for tscript in dict.keys():
        print(tscript)
        dict[tscript] = dict[tscript].replace('\n',' ')
        dict[tscript] = dict[tscript].replace('strong','')
        dict[tscript] = dict[tscript].translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        dict[tscript] = re.sub('\w*\d\w*', '', dict[tscript])
        
    return dict

def text_preprocessing_pipeline_2(dict):
    '''
    Input: a dictionary with preprocessed transcript text (phase 1)
    Output: a dictionary with preprocessed transcript text (phase 2)
    ''' 
    ## Tokenize text into words, check for spelling errors, remove stopwords, lemmatize, and remove people's names
    stop_words = set(stopwords.words('english'))
    
    lemmatizer = WordNetLemmatizer()
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['V'] = wordnet.VERB 
    tag_map['J'] = wordnet.ADJ
    tag_map['R'] = wordnet.ADV
    
    for tscript in dict.keys():
        print(tscript)
        dict[tscript] = nltk.word_tokenize(dict[tscript])
        
        def correct_spellings(text):
            spell = SpellChecker()
            corrected_text = []
            misspelled_words = spell.unknown(text)
            for word in text:
                if word in misspelled_words:
                    corrected_text.append(spell.correction(word))
                else:
                    corrected_text.append(word)
            return corrected_text

        dict[tscript] = correct_spellings(dict[tscript])
        
        dict[tscript] = [word for word in dict[tscript] if word.lower() not in stop_words]
        dict[tscript] = [lemmatizer.lemmatize(word.lower(), tag_map[tag[0]]) 
                                   for word, tag in pos_tag(dict[tscript])]
        dict[tscript] = [word for word in dict[tscript] if len(word) > 1]
        dict[tscript] = [word for word, tag in pos_tag(dict[tscript]) if tag!='NNP' or word=='subscriber' or word=='xbox']
        
    return dict

def remove_custom_stopwords_unigrams(dict):
    '''
    Input: a dictionary with correct spellings for transcript text
    Output: a dictionary with custom stopwords removed from transcript text
    '''
    frequent_words = ['quarter', 'revenue', 'microsoft', 'year', 'window', 'business', 'so', 'think', 'call', 'see',
                      'pa', 'go', 'earnings', 'question', 'fiscal', 'operator', 'billion', 'inc', 'like', 'also', 'look',
                      'good', 'come', 'well', 'get', 'say', 'make', 'right', 'chris', 'copyright', 'expect', 'use', 
                      'next', 'corporation', 'would', 'give', 'weve', 'saw', 'im', 'chief', 'officer', 'today', 'yes',
                      'investor', 'relation', 'release', 'thing', 'spglobalcommarketintelligence', 'could', 'lot', 'let',
                      'result', 'one', 'talk', 'really', 'want', 'million', 'thank', 'thanks', 'first', 'second', 'grow',
                      'growth', 'market', 'point', 'last', 'global', 'within', 'us', 'satya', 'across', 'line', 'point'
                      'even', 'up', 'include', 'cfo', 'overall', 'way', 'take', 'around', 'due', 'division', 'continue',
                      'liddle', 'server', 'presentation', 'welcome', 'jan', 'apr', 'jul', 'oct', 'third', 'fourth', 
                      'conference', 'instruction', 'my', 'turn', 'record', 'participant', 'colleen', 'healy', 'general',
                      'manager', 'bill', 'koefoed', 'may', 'sir', 'please', 'objection', 'disconnect', 'greeting',
                      'reminder', 'mike', 'spencer', 'pleasure', 'host', 'suh', 'proceed', 'afternoon', 'still', 'till',
                      'della', 'amy', 'hood', 'peter', 'klein', 'adam', 'cio', 'ceo', 'likely', 'it', 'hi', 'john',
                      'feel', 'much', 'wwwmicrosoftcommsft', 'alan', 'karl', 'ian', 'says', 'keith', 'difucci', 'steve',
                      'pc', 'sp', 'liddell', 'nadella', 'charlie', 'william', 'vice', 'president', 'ubs', 'bellini',
                      'holt', 'lync', 'fy', 'director', 'deutsche', 'keirstead', 'christopher', 'join', 'jason', 'frank',
                      'brod', 'michael', 'financial', 'increase', 'constant', 'currency', 'intelligence','former',
                      'charge', 'guarantee', 'three', 'tech', 'client','friar','senior','corporate','accounting','deputy',
                      'counsel','vp','six','five','four','sara','breza', 'db', 'ross', 'wei', 'seethoff','research',
                      'maguire', 'sarah', 'inaudible', 'mbd', 'japan', 'intelligent', 'xp', 'oppenheimer', 'ive', 'ag', 
                      'fx', 'thill', 'citigroup', 'egbert',"i've"]

    for tscript in dict.keys():
        print(tscript)
        dict[tscript] = [word for word in dict[tscript] if word not in frequent_words]
        
    return dict

def create_bigrams(dict): 
    '''
    Input: a dictionary with correct spellings for transcript text
    Output: a dictionary with bigrams in transcript text
    '''
    for tscript in dict.keys():
        print(tscript)
        dict[tscript] = [word1 + ' ' + word2 for word1, word2 in list(ngrams(dict[tscript], 2))]
        
    return dict   

def remove_custom_stopwords_bigrams(dict):
    '''
    Input: a dictionary with bigrams in transcript text
    Output: a dictionary with custom bigram stopwords removed from transcript text
    '''
    frequent_words_bi = ['former chief','peter klein','klein former','chris liddell','call oct','call jul','call jan',
                         'call apr','officer yes','full fiscal','former general','william koefoed','koefoed former',
                         'amy hood','hood executive','executive vp','vp cfo','point view','colleen healy',
                         'christopher liddell','year year','question please','division peter','bellini ubs','chris suh',
                         'please operator','spencer general','michael spencer','rbc capital','macmillan rbc',
                         'first quarter','fourth quarter','second quarter','third quarter','tech guarantee','satya nadella',
                         'sp global','constant currency','ceo director','nadella ceo','bill koefoed','microsoft business',
                         'business division','division revenue','increase constant','grow constant','business pc',
                         'currency drive','business process','quarter full','segment gross','personal computing',
                         'suh general','director yes','friar goldman','percentage point','currency gross','cloud gross',
                         'dollar increase','point yearoveryear','officer thanks','thill citigroup']

    for tscript in dict.keys():
        print(tscript)
        dict[tscript] = [word for word in dict[tscript] if word not in frequent_words_bi]
        
    return dict

def qa_text_processing_pipeline(dict):
    '''
    Input: a dictionary with names of docs as keys and Q&A portion of transcript text as values
    Output: a dictionary with preprocessed Q&A text
    ''' 
    stop_words = set(stopwords.words('english'))

    lemmatizer = WordNetLemmatizer()
    tag_map = defaultdict(lambda : wordnet.NOUN)
    tag_map['V'] = wordnet.VERB 
    tag_map['J'] = wordnet.ADJ
    tag_map['R'] = wordnet.ADV
    
    for tscript in dict.keys():
        print(tscript)
        ## Remove apostrophes, punctuations, and numbers
        dict[tscript] = dict[tscript].replace("'","")
        dict[tscript] = dict[tscript].replace("’","")
        dict[tscript] = dict[tscript].translate(str.maketrans('', '', string.punctuation))
        dict[tscript] = re.sub('\w*\d\w*', '', dict[tscript])
        
        ## Tokenize, remove stopwords, lemmatize
        dict[tscript] = nltk.word_tokenize(dict[tscript])
        dict[tscript] = [word for word in dict[tscript] if word.lower() not in stop_words]
        dict[tscript] = [lemmatizer.lemmatize(word.lower(), tag_map[tag[0]]) 
                                       for word, tag in pos_tag(dict[tscript])]
        dict[tscript] = [word for word in dict[tscript] if len(word) > 1]
        dict[tscript] = [word for word in dict[tscript] if word != 'ill' and word != 'question']
        
    return dict