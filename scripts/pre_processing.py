# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:58:11 2021

@author: Md. Prince
"""

from sklearn.datasets import load_files
import nltk,os
import re
import numpy as np
import pandas as pd
#nltk.download('punkt')#Download the Package only once
DATASET_CSV_FILE_PATH = os.curdir + "\\..\\dataset\\dataset.csv"
df = pd.read_csv(DATASET_CSV_FILE_PATH)




def pre_prossing(dataset):
    X = dataset['news']
    #print (X[0].decode('utf-8'))
    punct=""". ! ( ) - _ / < > ; : । ‘ " ’ , ? # @ $ % ^ & * = + { [ } ] \ | “ ” '"""
    punc=nltk.word_tokenize(punct)
    #stop_word=np.genfromtxt('stop2.txt')
    stop_words=""". ! ( ) - _ / < > ; : । ‘ " ’ , ? # @ $ % ^ & * = + { [ } ] \ | ' এ না করে এর থেকে এই আমি হবে আর জন্য যে আমার তার পর আছে এবং কি তাদের এটা কোনো এক একটা কিছু করা হয় করতে সে নেই কিন্তু তারা কথা তবে এখন বলে উপর মনে সাথে ১ এ যদি দিয়ে হলো সব বা মাঝে কাছে হয়ে মত আমাদের ও নিয়ে ছিলো তাই আগে যারা ২ করি করার যাবে উনি সেটা বেশি কেউ তখন অনেক যখন যায় শুধু ৩ হয় হয়েছে নি দিকে ঐ আমরা কোন থাকে যা যত করেন আপনার করবে উনার ভালো আমাকে তাকে আপনি পারে কারন বলেন আরো যেন কে আবার বলেছেন তোমাদের হচ্ছে দেয়া এখানে দিয়ে দিতে তোমরা তিনি বরং হলে সেই তুমি হয়ে বলা দেখে সবাই রা মানে নিজের হতে করেছেন থাকবে বললেন এমন জন তাহলে তো আল করলে নিয়ে করেছে ভাই তোমার গিয়েছে নাকি বের এগুলো করছে ছিল এরকম তা যার ব্যপারে কিনা যায় বলতে হয়েছে যেতে এসে এসেছে দেন যেমন এত যেটা যাচ্ছে একই করছি হতো নিজে এখনো করলাম খুব গত"""
    stop_word=nltk.word_tokenize(stop_words)
    dataset=[]
    for i in range(0,len(X)):
            words = nltk.word_tokenize(X[i])#.decode('utf-8'))#list
            X[i]=" "
            #docset=[]
            modified_words=[]
            for word in words:
                charcter = list(word)
                for char in charcter:
                    if char in punc:
                        charcter.remove(char)
                word=''.join(charcter)
                modified_words.append(word)
            
            for word in modified_words:
                
                if word in stop_word:
                    modified_words.remove(word)
            X[i]=' '.join(modified_words)
    return X
'''Here we remove puntuation and stop words from test dataset
 and then we store the result in to a new .csv file'''
news0 = pre_prossing(df)
df['news'] = news0
df.to_csv('pre_processed_data.csv', index=False)


