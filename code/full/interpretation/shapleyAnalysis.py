import pickle
import shap
import datasets
import scipy as sp
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys

#for collapser
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltkStopwords = set(stopwords.words('english'))

## PREDICTION FUNCTION ##

#returns pre-softmax log probabilities for all classes
def getClassificationLogOdds(sampleInput,model,tokenizer_,tokenizer_max_length,device, mask_all = False):

    tokens = tokenizer_.encode_plus(sampleInput, max_length = tokenizer_max_length,
                                            padding="max_length",
                                            truncation=True)
    
    if mask_all: 
        masked = tokens.input_ids.copy()
        for ni,i in enumerate(tokens.input_ids):
            if i>4: masked[ni] = 4
        tokens['input_ids'] = masked
    
    formatSample = lambda x: torch.tensor(x).unsqueeze(0).to(device)

    input_ids = formatSample(tokens["input_ids"])
    mask = formatSample(tokens["attention_mask"])
    
    model.eval()
    with torch.no_grad(): _, _ , pred = model(input_ids, mask)
    probs = pred.to('cpu').numpy()[0]
    return probs

## SHAP VALUES ##
def getShapValues_classification(sentences, model, tokenizer_, tokenizer_max_length,device):
    predictor = lambda sampleInputs: [getClassificationLogOdds(sentence,model,tokenizer_,tokenizer_max_length,device) for sentence in sampleInputs]
    explainer = shap.Explainer(predictor, tokenizer_)
    shap_values = explainer(sentences) 
    return shap_values

def getShapValues(predictFunction, sentences, model, tokenizer_, tokenizer_max_length, device):
    predictor = lambda sampleInputs: [predictFunction(sentence,model,tokenizer_,tokenizer_max_length,device) for sentence in sampleInputs]
    explainer = shap.Explainer(predictor, tokenizer_)
    shap_values = explainer(sentences) 
    return shap_values

## SAVE AND LOAD ##
def ensureDirectoryExists(dir_name):
    assert dir_name[-1] == '/', 'directory name must end with /'
    if os.path.exists(dir_name):
        exists=True
        if os.path.isdir(dir_name)==False:
            assert False,'Non-directory file '+dir_name+' exists.'
    else: 
        exists=False
        os.makedirs(dir_name)
    return exists

def pickleLoad(path):
    if os.path.exists(path): 
        with open(path,'rb') as f: data=pickle.load(f)
        print('loading data from',path)
        return data
    else: 
        print('Load failed. File does not exist:', path)
        return False
    
def pickleSave(data,dir_name,fileName,overwrite=True):
    path=dir_name+fileName
    if os.path.exists(path): 
        if overwrite: saveData=True ## FILE EXISTS, OVERWRITE
        else: saveData=False ##FILE EXISTS, DO NOT OVERWRITE
    else: saveData=True ##FILE DOES NOT EXIST
    if saveData: 
        ensureDirectoryExists(dir_name)
        with open(path,'wb') as f: pickle.dump(data,f)
        print('saved to',path)

## ANALYSIS OF SHAP VALUES ##

# note: words are defined by contiguous characters separated by spaces
# This means that when we aggregate shap values of unique words, words with punctuation are unique, e.g. "protein." and "protein"
# The next step is to collapse keys which differ by punctuation, capitalization, and plurality, e.g. "protein.", "(protein", "Proteins)", etc. all become "protein"
# In collapsing, shap values of punctuation are included in the sum of that word's tokens

# Pass a single shap value object, i.e. shap_values[0]
# Returns a dictionary of shap values grouped by word (key = word, value = list of shap values of tokens in the word)
def groupShapValuesByWord(shap_vals,shap_data,verbose=False):
    sentence = ''.join(shap_data)
    filterOut=['', ' '] #exclude empty strings and spaces from tokens, words, and shap values

    #align chars, tokens, and words     
    words = [i for i in sentence.split(' ') if i not in filterOut]              # ex. words =         ['aab','cdd','ee']
    tokens = [t for t in shap_data if t not in filterOut]                       # ex. tokens =        ['aa', 'b', 'c', 'dd', 'ee']
    tokenLens = [len(t.strip()) for t in tokens if t not in filterOut]          # ex. tokenLens =     [2, 1, 1, 2, 2] <- sum of previous lengths at each token gives the index of the last char of the token
    wordLens = [len(w) for w in words]                                          # ex. wordLens =      [3, 3, 2] <- used to create charWordIndex
    charWordIndex = [nj for nj,j in enumerate(wordLens) for i in range(j)]      # ex. charWordIndex = [0, 0, 0, 1, 1, 1, 2, 2] <- index of the word that the char belongs to, i.e., 'b' belongs to the first word 'aab'
    if verbose: 
        print(sentence)
        print(words)
        print(tokens)

    filtered_shap_values = [(shap_vals[i],shap_data[i]) for i in range(len(shap_vals)) if shap_data[i] not in filterOut] # remove shap values of tokens that are filtered out

    # group shap values of tokens which belong to the same word (key = word, value = list of shap values of tokens in the word)
    shap_values_by_word = {}
    for i in range(len(tokenLens)):

        token = tokens[i]
        charIdx = sum(tokenLens[:i]) 
        wordIdx = charWordIndex[charIdx] 
        word = words[wordIdx] 

        if shap_values_by_word.get(word) is None:
            shap_values_by_word[word] = []

        tokenIsGroupedWithLastToken = False
        if i>0:
            lastTokenCharIdx = sum(tokenLens[:i-1])
            lastWordIdx = charWordIndex[lastTokenCharIdx]
            tokenIsGroupedWithLastToken = wordIdx==lastWordIdx

            if verbose: print('grouped with last token:',tokenIsGroupedWithLastToken,'\t', tokens[i-1],token)

        if tokenIsGroupedWithLastToken:
            shap_values_by_word[word][-1].append(filtered_shap_values[i][0])
            if verbose: print(word,':',shap_values_by_word[word])
        else:   
            shap_values_by_word[word].append([filtered_shap_values[i][0]])
            if verbose: print(word,':',shap_values_by_word[word]) 
        assert token in filtered_shap_values[i][1], str(token)+' '+str(filtered_shap_values[i][1]) #make sure the token in the list belongs to the token in the input shapValues
        
        if verbose: print(word,token,filtered_shap_values[i],'\n')

    return shap_values_by_word

#returns a list of dicts of shap values grouped by word for each sample in shap_values
# mimics shape of shap_values; for each sample, .values is replaced by dict values, and .data is replaced by dict keys
def getShapValuesGroupedByWord(shap_values):
    shap_data,shap_vals = shap_values.data,shap_values.values
    shap_values_by_word = []
    for i in range(len(shap_data)):
        shap_values_by_word.append(groupShapValuesByWord(shap_vals[i],shap_data[i]))
    return shap_values_by_word

# returns
# token_analysis: a list with a dict for each class; each dict contains - for each token (key), a list of all shap values for each instance of that token
# token_analysis_indexes: for each token (key), a list of indexes corresponding to the gene's index in shap_values (which should be the same index in gene_loaded_data)
def getShapValuesAnalysisAndIndexes(shap_values, indexList = None):
    shap_data,shap_vals = shap_values.data,shap_values.values
    nClasses = len(shap_vals[0][0])
    token_analysis=[{} for i in range(nClasses)]
    token_analysis_indexes={}
    for n,tokens in enumerate(shap_data):
        for tokenIndex,token in enumerate(tokens):
            for c in range(nClasses):
                if token_analysis[c].get(token) is None: token_analysis[c][token]=[]
                token_analysis[c][token].append(shap_vals[n][tokenIndex][c])
            if token_analysis_indexes.get(token) is None: token_analysis_indexes[token]=[]
            index = n if indexList is None else indexList[n]
            token_analysis_indexes[token].append(index)
    return token_analysis, token_analysis_indexes

# returns
# word_analysis: a list with a dict for each class; each dict contains - for each word (key), a list of all shap values for each instance of that word (with a function performed on the shap values of each words tokens)
# word_analysis_indexes: for each word (key), a list of indexes corresponding to the gene's index in shap_values (which should be the same index in gene_loaded_data)
def getShapValuesAnalysisAndIndexes_groupedByWordFunction(shap_values_grouped_by_word,indexList,f=np.sum):
    # shap_values_grouped_by_word: for each summary, a dict of words (keys) and lists of lists of shap values (values) for each instance the word appears in the summary
    # example entry: 
    # 'Gproteins': [[array([ 0.322, -0.285]), array([ 0.110, -0.102]), array([-0.022, -0.003])],
    #               [array([ 0.189, -0.261]), array([ 0.090, -0.091]), array([-0.032, -0.002])]]
    nClasses = len(list(shap_values_grouped_by_word[0].values())[0][0][0])
    word_analysis=[{} for i in range(nClasses)]
    word_analysis_indexes={}
    for n,gene in enumerate(shap_values_grouped_by_word):
        for word,vals in gene.items():
            for instance in vals:
                for c in range(nClasses):
                    if word_analysis[c].get(word) is None: word_analysis[c][word]=[]
                    fVals = f([val[c] for val in instance])
                    word_analysis[c][word].append(fVals)
                if word_analysis_indexes.get(word) is None:
                    word_analysis_indexes[word]=[]
                index = n if indexList is None else indexList[n]
                word_analysis_indexes[word].append(index)
    return word_analysis, word_analysis_indexes  

getShapValuesAnalysisAndIndexes_groupedByWordSum = lambda x, indexList: getShapValuesAnalysisAndIndexes_groupedByWordFunction(x,indexList,f=np.sum)
getShapValuesAnalysisAndIndexes_groupedByWordMax = lambda x, indexList: getShapValuesAnalysisAndIndexes_groupedByWordFunction(x,indexList,f=max)                               

def getXPercentile(X,vals,gene_indexes):
    vals = np.array(vals)
    vals = np.sort(vals)
    vals_indexes = np.argsort(vals)
    gene_indexes = [gene_indexes[i] for i in vals_indexes]
    percentile_value = np.percentile(vals,X)
    index_of_X_percentile = np.argmin(np.abs(vals-percentile_value))
    gene_index_of_X_percentile = gene_indexes[index_of_X_percentile]
    return percentile_value, gene_index_of_X_percentile

# pass shap values analysis with a single array per token (i.e. for a single class)
# returns analysis with the median of the top X percentile (value) for each token (key)
def getXPercentileAnalysis(shap_values_analysis,indexes,X):
    percentileAnalysis = {}
    percentileAnalysis_indexes = {}
    for token,shapvals in shap_values_analysis.items():
        val,idx = getXPercentile(X,shapvals,indexes[token])
        percentileAnalysis[token] = val
        percentileAnalysis_indexes[token] = idx
    return percentileAnalysis, percentileAnalysis_indexes

# filter out keys with less than minCount values (only for dicts with single lists as values)
def filterAnalysisByNumberOfOccurances(shap_values_analysis,minCount):
    filteredAnalysis = shap_values_analysis.copy()
    for token,vals in shap_values_analysis.items():
        if len(vals) < minCount: filteredAnalysis.pop(token)
    return filteredAnalysis

def flattenWord(word):
    chars_to_remove = string.punctuation
    table = str.maketrans('', '', chars_to_remove)
    return word.lower().strip().translate(table)

# collapse keys that differ only by punctuation, capitals, or spaces 
# optionally collapse plurals; this works by searching if a non-plural version exists in the set of keys and collapsing if this is true
# optionally filter out words less than minLength
# optionally filter out a list of words (stops), meant for nltk stopwords
# optionally remove any words that occur less than minCount times
def collapseShapValues(shap_values_analysis,shap_values_analysis_indexes=None,collapsePlural=False,minStringLength=1,maxStringLength=15,stops=None, minCount=0):
    shap_values_analysis_collapsed = {}
    shap_values_analysis_indexes_collapsed={}
    chars_to_remove = string.punctuation
    table = str.maketrans('', '', chars_to_remove)
    collapse = lambda x : x.lower().strip().translate(table)
    collapsedSet = set([collapse(token) for token in shap_values_analysis.keys()])
    if collapsePlural: pluralCount=0
    for token,val in shap_values_analysis.items():
        collapsedToken = collapse(token)

        #check for empty string, minStringLength, stopwords
        if collapsedToken == '': continue
        if len(collapsedToken)<minStringLength: continue
        if (stops is not None) and (collapsedToken in stops): continue
        if len(collapsedToken)>maxStringLength: continue

        #check if non-plural version of the token exists in the set of collapsed keys
        if collapsePlural:
            if (collapsedToken[-1]=='s') and (collapsedToken.rstrip('s') in collapsedSet): #non-plural version of the token exists in the collapsedSet
                # print(collapsedToken,'is plural')
                pluralCount+=1
                collapsedToken = collapsedToken.rstrip('s')
        
        if shap_values_analysis_collapsed.get(collapsedToken) is None: shap_values_analysis_collapsed[collapsedToken] = val
        else: shap_values_analysis_collapsed[collapsedToken].extend(val)
        if shap_values_analysis_indexes is not None:
            if shap_values_analysis_indexes_collapsed.get(collapsedToken) is None: shap_values_analysis_indexes_collapsed[collapsedToken] = shap_values_analysis_indexes[token]
            else: shap_values_analysis_indexes_collapsed[collapsedToken].extend(shap_values_analysis_indexes[token])

    shap_values_analysis_collapsed = filterAnalysisByNumberOfOccurances(shap_values_analysis_collapsed,minCount)    

    if collapsePlural: print(str(pluralCount),'plurals found, out of',str(len(shap_values_analysis.keys())),'tokens (',str(round(pluralCount/len(shap_values_analysis.keys())*100,2))+'% )')
    print(type(shap_values_analysis_collapsed))
    return shap_values_analysis_collapsed, shap_values_analysis_indexes_collapsed

## GET SHAP VALUES AND AGGREGATE INTO DICTIONARY OF TOKENS/WORDS ##
def getSHAPValues(sentences, model, tokenizer_, tokenizer_max_length, device, dataset_name, save_dir):
    ensureDirectoryExists(save_dir)
    print('generating shapley explanations')
    shap_values = pickleLoad(save_dir+dataset_name+'.pkl')
    if shap_values is False:
        shap_values = getShapValues_classification(sentences,model,tokenizer_,tokenizer_max_length,device)
        pickleSave(shap_values,save_dir, dataset_name+'.pkl')
    return shap_values

# token_analysis: key = token, value = list of shap values across all instances
# token_analysis_indexes: key = token, value = list of indexes in token_analysis corresponding to the gene's index in shap_values input
# word_sum_analysis: key = token, value = list of shap values across all instances, summed over all tokens in the word
# word_sum_analysis_indexes: key = token, value = list of indexes in word_sum_analysis corresponding to the gene's index in shap_values input
# shap_vals_by_word: key = word, value = list of lists; for each occurance of the word, a list of shap values of the tokens composing it
def getSHAPAnalysis(params):
    shap_values = params['shap_values']
    indexList = params['indexList'] if 'indexList' in params.keys() else None
    print('Analyzing SHAP Values\n')
    token_analysis, token_analysis_indexes = getShapValuesAnalysisAndIndexes(shap_values, indexList)
    shap_vals_by_word = getShapValuesGroupedByWord(shap_values) # (key = word, value = list of shap values of tokens in the word)
    word_sum_analysis, word_sum_analysis_indexes = getShapValuesAnalysisAndIndexes_groupedByWordSum(shap_vals_by_word, indexList)
    print('''returning:
    token_analysis:             key = token     value = list of shap values across all instances
    token_analysis_indexes:     key = token     value = list of indexes in token_analysis corresponding to the gene's index in shap_values input
    word_sum_analysis:          key = word      value = list of shap values across all instances, summed over all tokens in the word
    word_sum_analysis_indexes:  key = word      value = list of indexes in word_sum_analysis corresponding to the gene's index in shap_values input
    shap_vals_by_word:          key = word      value = list of lists; for each occurance of the word, a list of shap values of the tokens composing it''')
    return {'token_analysis':token_analysis,'token_analysis_indexes':token_analysis_indexes,'word_sum_analysis':word_sum_analysis,'word_sum_analysis_indexes':word_sum_analysis_indexes,'shap_vals_by_word':shap_vals_by_word}

def getCollpasedAnalysis(params):
    analysis = params['analysis'] 
    kind='word_sum' if 'kind' not in params.keys() else params['kind']
    minOccurences = 1 if 'minOccurences' not in params.keys() else params['minOccurences']
    maxStringLength = 15 if 'maxStringLength' not in params.keys() else params['maxStringLength']
    collapsePlural = True if 'collapsePlural' not in params.keys() else params['collapsePlural']
    minStringLength = 1 if 'minStringLength' not in params.keys() else params['minStringLength']
    stops = nltkStopwords if 'stops' not in params.keys() else params['stops']
    print('COLLAPSING AND FILTERING SHAP VALUES')
    print('collapsePlural:',collapsePlural)
    print('minStringLength:',minStringLength)
    print('maxStringLength:',maxStringLength)
    print('filtered words:',len(stops))
    print('minOccurences:',minOccurences)
    
    analysis_=analysis[f'{kind}_analysis']
    analysis_indexes_=analysis[f'{kind}_analysis_indexes']
    
    collapsed_analyses = []
    
    for i in analysis_: #i for each class
        collapsed_analysis,collapsed_analysis_indexes = collapseShapValues(i,shap_values_analysis_indexes=analysis_indexes_,collapsePlural=collapsePlural,maxStringLength = maxStringLength,minStringLength=minStringLength,stops=stops, minCount=minOccurences)
        collapsed_analyses.append(collapsed_analysis)

    ret={}
    ret['collapsed_'+kind+'_analysis'] = collapsed_analyses
    ret['collapsed_'+kind+'_analysis_indexes'] = collapsed_analysis_indexes  

    print('returning:')
    for i in ret.keys():
        print('\t',i)
    return ret
    

def getPercentileAnalysis(params):
    analysis=params['analysis']
    percentile=params['percentile']
    kind='collapsed_word_sum' if 'kind' not in params.keys() else params['kind']

    print('GETTING PERCENTILE VALUES', percentile)
    vals = analysis[f'{kind}_analysis'] 
    vals_indexes = analysis[f'{kind}_analysis_indexes']

    percentile_analysis = {f'{kind}_sorted_{percentile}_percentile_vals':[],
           f'{kind}_sorted_{percentile}_percentile_keys':[],
           f'{kind}_sorted_{percentile}_percentile_indexes':[],
           f'{kind}_sorted_{percentile}_percentile_counts':[],
           f'{kind}_sorted_{percentile}_percentile_log_counts':[]}
    
    for classIndex,i in enumerate(vals):
        percentileAnalysis, percentileAnalysis_indexes = getXPercentileAnalysis(i,vals_indexes,percentile)

        sorted_percentile_vals = sorted(percentileAnalysis.values(), reverse=True)
        sorted_percentile_keys = sorted(percentileAnalysis.keys(), key=lambda x: percentileAnalysis[x], reverse=True)
        sorted_percentile_indexes = [percentileAnalysis_indexes[key] for key in sorted_percentile_keys]
        sorted_percentile_counts = [len(vals[classIndex][key]) for key in sorted_percentile_keys]
        sorted_percentile_log_counts = np.log10(sorted_percentile_counts)

        #these must be in the same order as percentile_analysis.keys()
        allvals = sorted_percentile_vals, sorted_percentile_keys, sorted_percentile_indexes, sorted_percentile_counts, sorted_percentile_log_counts
        for key,val in zip(percentile_analysis.keys(),allvals):
            percentile_analysis[key].append(val)

    print('returning:')
    for i in percentile_analysis.keys():
        print('\t',i)
    
    return percentile_analysis

# def generatePlots(analysis, nToPlot = 100, percentile=90, minOccurences = 10, collapsePlural = True, minStringLength = 1, stops = nltkStopwords, saveName=None):
#     print('COLLAPSING AND FILTERING SHAP VALUES')
#     print('collapsePlural:',collapsePlural)
#     print('minStringLength:',minStringLength)
#     print('filtered words:',len(stops))
#     print('percentile:',percentile)
#     print('minOccurences:',minOccurences)
#     collapsedValues= []
#     for ni,i in enumerate(analysis):
#         print('class',ni)
#         collapsedShapValues,_ = collapseShapValues(i,collapsePlural=collapsePlural,minStringLength=minStringLength,stops=stops)
#         collapsedValues.append(collapsedShapValues)
#     filteredValues = []
#     for i in collapsedValues:
#         filteredValues.append(filterAnalysisByNumberOfOccurances(i,minOccurences))
#     percentileValuesAnalyses = []
#     for i in filteredValues:
#         percentileValuesAnalyses.append(getXPercentileAnalysis(i,percentile))
    
#         # # PLOT TOP X PERCENTILE SHAP VALUES
#     for classIndex,percentileValues in enumerate(percentileValuesAnalyses):
#         plt.figure(figsize=(20,5))
#         plt.title('Class %1d, %2dth Percentile Shap Value'%(classIndex,percentile))
#         top_vals = sorted(percentileValues.values(),reverse=True)[:nToPlot]
#         top_keys = sorted(percentileValues.keys(),key=lambda x: percentileValues[x],reverse=True)[:nToPlot]
#         plt.bar(range(nToPlot),top_vals)
#         plt.xticks(range(nToPlot),top_keys,rotation=90)
#         plt.grid(axis='y')
#         # plt.ylim(0,1.25)
#         # plt.yticks(np.arange(0,1.2,0.1))
#         plt.show()
#         if saveName is not None: plt.savefig(saveName+'_barPlot_class_'+str(classIndex)+'_top_'+str(nToPlot)+'_percentile.png')

#         #scatter plot token counts vs XPercentile_grouped_by_word_membrane
#         plt.figure(figsize=(20,5))
#         plt.title('Class %1d, Token Counts vs %2dth Percentile Shap Value, Top %d'%(classIndex,percentile,nToPlot))
#         top_counts = [len(filteredValues[classIndex][key]) for key in top_keys]
#         top_logCounts = np.log10(top_counts)
#         plt.scatter(top_logCounts,top_vals)
#         plt.xlabel('Log Token Counts')
#         plt.ylabel('%2dth Percentile Shap Value'%(percentile))
#         plt.grid(axis='y')
#         # plt.ylim(0,1.25)
#         # plt.yticks(np.arange(0,1.2,0.1))
#         #label points with token name
#         for i in range(len(top_keys)):
#             plt.annotate(top_keys[i],(top_logCounts[i],top_vals[i]))
#         plt.show()
#         if saveName is not None: plt.savefig(saveName+'_scatterPlot_class_'+str(classIndex)+'_top_'+str(nToPlot)+'_percentile.png')
    
    
    