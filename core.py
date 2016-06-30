#!/usr/bin/python3
# -*- coding: utf-8 -*-
import random
import nltk
import time
import NaiveBayesImp
import codecs
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier
import sys

class Core:
    #units = None
    #def __init__(self, label_probdist, feature_probdist):
    #   super().__init__(self, label_probdist, feature_probdist)
    documents = None
    classifier = None

    all = 0
    pos = 0
    neg = 0

    def extract_line(filename):
        return [line.rstrip('\n')
                for line in open("corpus/{0}".format(filename),encoding="utf-8")]


    def extract_word(line):
        return line.split()

    def extract_word_lower(line):
        lowers = []
        for s in line.split():
            lowers.append(s.lower())
        return lowers

    def filter_words(document):
        words = []
        for content in document:
            for word in content[0]:
                words.append(word)
        return words

    def filter_document(document):
        for i in range(0,5):
            document.pop(0)
        return document

    def merge_meta(document, meta):
        md = Core.filter_document(Core.extract_line(meta))
        index = 0
        for m in md:
            document[index] = tuple(m if i==1 else x
                                    for i, x in enumerate(document[index]))
            index += 1
            # if m == '1':
            #     Core.pos += 1
            # else:
            #     Core.neg += 1

        Core.all = index
        del index
        return document

    def document_features(document, units):
        document_words = set(document)
        features = {}
        for word in units:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    @staticmethod
    def process(train = 20, feature = 25, info = False):
        start = time.time()
        Core.documents = [(list(Core.extract_word_lower(line)),0)
                     for line in Core.extract_line('gogo_lhagvasuren_01.data')]
        Core.documents = Core.merge_meta(Core.filter_document(Core.documents), 'gogo_lhagvasuren_01.meta')

        random.shuffle(Core.documents)

        units = nltk.FreqDist(w for w in Core.filter_words(Core.documents))

        featuresets = [(Core.document_features(d,units),c) for (d,c) in Core.documents]

        size = int(len(featuresets) * 0.5)
        train_set = featuresets[:size]
        print("train set: "+str(size))
        test_set = featuresets[size:]
        print("test set: "+str(len(featuresets)-size))
        Core.classifier = nltk.NaiveBayesClassifier.train(train_set)

        for content,category in test_set:
            result = Core.classifier.classify(content)
            if result == category:
                #print("Corrent")
                Core.pos += 1
            else:
                #print("Wrong")
                Core.neg += 1

        print("\n---accuracy---")
        print("Correct: "+str(Core.pos)+", Negative: "+str(Core.neg))
        print(nltk.classify.accuracy(Core.classifier,test_set))
        print("-accuracy-end-\n")
        end = time.time()
        print("Time: "+str(end - start)+"\n")
        if info:
            return Core.classifier.show_most_informative_features(feature)
            #return NaiveBayesImp.show_most_informative_features(feature)

    @staticmethod
    def validator(param):
        test = False
        if Core.classifier == None:
            Core.process(20,25,False)
        else:
            if test:
                for content,category in param:
                    result = Core.classifier.classify(content)
                    if result == category:
                        result = 1
                    else:
                        result = 0
            else:
                result = Core.classifier.classify(param)
        return result

Core.process(98,10,True)
# for line in dump:
#     print(line)
