# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 01:31:45 2021

@author: Md. Prince
"""

#Import the Module
import time
start_time = time.time()
import pandas as pd
import numpy as np
import nltk
#nltk.download('punkt')#Download the Package
import os,csv

FEATURE_LIST_CSV_FILE_PATH = os.curdir + "\\..\\features\\features.csv"
PREPROCESSED_DATASET_CSV_FILE_PATH = os.curdir + "\\..\\dataset\\pre_processed_data.csv"

def find_unique(dataset,l):
    
    """
    This function will print the summary of the reviews and words distribution in the dataset. 
    
    Args:
        dataset: list of cleaned sentences   
        
    Returns:
        Number of documnets per class: int 
        Number of words per class: int
        Number of unique words per class: int
    """
    documents = []
    words = []
    u_words = []
    label_unique_words = []
    total_u_words = [word.strip().lower() for t in list(dataset.news) for word in t.strip().split()]
    class_label= [k for k,v in dataset.label.value_counts().to_dict().items()]
  # find word list
    for label in class_label: 
        if label == l:
            word_list = [word.strip().lower() for t in list(dataset[dataset.label==label].news) for word in t.strip().split()]
            counts = dict()
            for word in word_list:
                    counts[word] = counts.get(word, 0)+1
            # sort the dictionary of word list  
            ordered = sorted(counts.items(), key= lambda item: item[1],reverse = True)
            for k,v in ordered[:100]:
                #print("{}\t{}".format(k,v))
                label_unique_words.append(k)
    return label_unique_words


def economics_unique_words_counter(words,economicsUniqueWord_data):
    economicsUniqueWord_data_count = 0
    for word in words:
        for i in range(0,len(economicsUniqueWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word == economicsUniqueWord_data[i]:
                economicsUniqueWord_data_count = economicsUniqueWord_data_count+1
    return economicsUniqueWord_data_count

def entertainment_unique_words_counter(words,entertainmentUniqueWord_data):
    entertainmentUniqueWord_data_count = 0
    for word in words:
        for i in range(0,len(entertainmentUniqueWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word == entertainmentUniqueWord_data[i]:
                entertainmentUniqueWord_data_count = entertainmentUniqueWord_data_count+1
    return entertainmentUniqueWord_data_count

def international_unique_words_counter(words,internationalUniqueWord_data):
    internationalUniqueWord_data_count = 0
    for word in words:
        for i in range(0,len(internationalUniqueWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word == internationalUniqueWord_data[i]:
                internationalUniqueWord_data_count = internationalUniqueWord_data_count+1
    return internationalUniqueWord_data_count

def scienceAndTechnology_unique_words_counter(words,scienceAndTechnologyUniqueWord_data):
    scienceAndTechnologyUniqueWord_data_count = 0
    for word in words:
        for i in range(0,len(scienceAndTechnologyUniqueWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word == scienceAndTechnologyUniqueWord_data[i]:
                scienceAndTechnologyUniqueWord_data_count = scienceAndTechnologyUniqueWord_data_count+1
    return scienceAndTechnologyUniqueWord_data_count

def sports_unique_words_counter(words,sportsUniqueWord_data):
    sportsUniqueWord_data_count = 0
    for word in words:
        for i in range(0,len(sportsUniqueWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word == sportsUniqueWord_data[i]:
                sportsUniqueWord_data_count = sportsUniqueWord_data_count+1
    return sportsUniqueWord_data_count

def economics_related_words_counter(words,economicsRelatedWord_data):
    economicsRelatedWord_data_count = 0
    for word in words:
        for i in range(0,len(economicsRelatedWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word == economicsRelatedWord_data[i]:
                economicsRelatedWord_data_count = economicsRelatedWord_data_count+1
    return economicsRelatedWord_data_count

def entertainment_related_words_counter(words,entertainmentRelatedWord_data):
    entertainmentRelatedWord_data_count = 0
    for word in words:
        for i in range(0,len(entertainmentRelatedWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word == entertainmentRelatedWord_data[i]:
                entertainmentRelatedWord_data_count = entertainmentRelatedWord_data_count+1
    return entertainmentRelatedWord_data_count

def scienceAndTechnology_related_words_counter(words,scienceAndTechnologyRelatedWord_data):
    scienceAndTechnologyRelatedWord_data_count = 0
    for word in words:
        for i in range(0,len(scienceAndTechnologyRelatedWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word ==  scienceAndTechnologyRelatedWord_data[i]:
                scienceAndTechnologyRelatedWord_data_count = scienceAndTechnologyRelatedWord_data_count+1
    return scienceAndTechnologyRelatedWord_data_count

def sports_related_words_counter(words,sportsRelatedWord_data):
    sportsRelatedWord_data_count = 0
    for word in words:
        for i in range(0,len(sportsRelatedWord_data)):
            #print(f'News Word: {word} \t Unique Word:{economicsUniqueWord_data[i]}')
            if word == sportsRelatedWord_data[i]:
                sportsRelatedWord_data_count = sportsRelatedWord_data_count+1
    return sportsRelatedWord_data_count

def main_feature_extraction(dataset):
        print(dataset.shape)
        news = dataset['news'].values
        label = dataset['label'].values
        related_word_dataset = pd.read_csv('RelatedWordNew.csv')
        #Unique Words From New
        economicsUniqueWord_data = find_unique(dataset,0)
        entertainmentUniqueWord_data = find_unique(dataset,1)
        internationalUniqueWord_data = find_unique(dataset,2)
        scienceAndTechnologyUniqueWord_data = find_unique(dataset,3)
        sportsUniqueWord_data = find_unique(dataset,4)
        economicsRelatedWord_data = related_word_dataset['EconomicsRelatedWord'].dropna().values
        entertainmentRelatedWord_data = related_word_dataset['EntertainmentRelatedWord'].dropna().values
        scienceAndTechnologyRelatedWord_data = related_word_dataset['ScienceAndTechnologyRelatedWord'].dropna().values
        sportsRelatedWord_data = related_word_dataset['SportsRelatedWord'].dropna().values
        
        
    # =============================================================================
    # Unique Words based features
    # =============================================================================
        economics_unique_words_count = []
        entertainment_unique_words_count = []
        international_unique_words_count = []
        scienceAndTechnology_unique_words_count = []
        sports_unique_words_count = []
        economics_related_words_count = []
        entertainment_related_words_count = []
        scienceAndTechnology_related_words_count = []
        sports_related_words_count = []
        
        for i in range(0,len(news)):
            words = nltk.word_tokenize(news[i])
            economics_unique_words_count.insert(i,economics_unique_words_counter(words,economicsUniqueWord_data))
            entertainment_unique_words_count.insert(i,entertainment_unique_words_counter(words,entertainmentUniqueWord_data))
            international_unique_words_count.insert(i,international_unique_words_counter(words, internationalUniqueWord_data))
            scienceAndTechnology_unique_words_count.insert(i,scienceAndTechnology_unique_words_counter(words, scienceAndTechnologyUniqueWord_data))
            sports_unique_words_count.insert(i,sports_unique_words_counter(words, sportsUniqueWord_data))
            economics_related_words_count.insert(i,economics_related_words_counter(words,economicsRelatedWord_data))
            entertainment_related_words_count.insert(i,entertainment_related_words_counter(words,entertainmentRelatedWord_data))
            scienceAndTechnology_related_words_count.insert(i,scienceAndTechnology_related_words_counter(words,scienceAndTechnologyRelatedWord_data))
            sports_related_words_count.insert(i,sports_related_words_counter(words,sportsRelatedWord_data))
        
        # combine all features together to produce features file
        feature_label = zip(label,
                            economics_unique_words_count,
                            entertainment_unique_words_count,
                            international_unique_words_count,
                            scienceAndTechnology_unique_words_count,
                            sports_unique_words_count,
                            economics_related_words_count,
                            entertainment_related_words_count,
                            scienceAndTechnology_related_words_count,
                            sports_related_words_count
                            )
        headers = ['Label',
                   'EconomicsUniqueWordCount',
                   'EntertainmentUniqueWordCount',
                   'InternationalUniqueWordCount',
                   'ScienceAndTechnologyUniqueWordCount',
                   'SportsUniqueWordCount',
                   'EconomicsRelatedWordCount',
                   'EntertainmentRelatedWordCount',
                   'ScienceAndTechnologyRelatedWordCount',
                   'SportsRelatedWordCount'
            ]
        if dataset.shape[0]<=500:
            # Writing headers to the new .csv file
            with open(FEATURE_LIST_CSV_FILE_PATH+'test_features.csv', "w", newline='') as header:
                header = csv.writer(header)
                header.writerow(headers)
        
            # Append the feature list to the file
            with open(FEATURE_LIST_CSV_FILE_PATH+'test_features.csv', "a", newline='') as feature_csv:
                writer = csv.writer(feature_csv)
                for line in feature_label:
                    writer.writerow(line)
        elif dataset.shape[0]>500:
            # Writing headers to the new .csv file
            with open(FEATURE_LIST_CSV_FILE_PATH+'train_features.csv', "w", newline='') as header:
                header = csv.writer(header)
                header.writerow(headers)
        
            # Append the feature list to the file
            with open(FEATURE_LIST_CSV_FILE_PATH+'train_features.csv', "a", newline='') as feature_csv:
                writer = csv.writer(feature_csv)
                for line in feature_label:
                    writer.writerow(line)



#This is the main function
def main():
    #test_dataset = pd.read_csv('pre_processed_test_data.csv')
    pre_processed_data = pd.read_csv(PREPROCESSED_DATASET_CSV_FILE_PATH)
    #main_feature_extraction(test_dataset)
    main_feature_extraction(pre_processed_data)


if __name__ == "__main__":
    main()
#to add new line
print()
print("Features has been created successfully.")
#calculate execution time
end_time = time.time() - start_time
total_minutes = int(end_time)/60
hours = total_minutes/60
minutes = total_minutes%60
seconds = int(end_time)%60
print("--- %d Hours %d Minutes %d Seconds ---" % (hours, minutes, seconds))