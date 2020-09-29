import collections
import sys
import getopt
import os
import math
import re
import operator
import csv




class NaiveBayes:
  class TrainSplit:
    """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
    """
    def __init__(self):
      self.train = []
      self.test = []

  class Example:
    """Represents a document with a label. klass is 'pos' or 'neg' by convention.
       words is a list of strings.
    """
    def __init__(self):
      self.klass = ''
      self.words = []


  def __init__(self):
    """NaiveBayes initialization"""
    self.FILTER_STOP_WORDS = False
    self.BOOLEAN_NB = False
    self.BEST_MODEL = False
    self.stopList = set(self.readFile('data/english.stop'))
    self.numFolds = 10

    self.posText = {}   # megatext for pos review, frequency
    self.negText = {}   # megatext for neg reviews, frequency
    self.text = {}     # megatext for all reviews
    self.countPosWords = 0.0    # total number of words in positve megatext
    self.countNegWords = 0.0    # total number of words in negatvie megatext
    self.countPosReviews = 0.0    # number of positive reviews
    self.countNegReviews = 0.0    # number of negative reviews
    
    self.train_vocab = {}
    self.shared_vocab = {}
    self.word_ratios = {}
    self.negative_words = []


    


  #############################################################################
  # TODO TODO TODO TODO TODO 
  # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
  # Boolean (Binarized) features.
  # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
  # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
  # that relies on feature counts.
  #
  # If the BEST_MODEL flag is true, include your new features and/or heuristics that
  # you believe would be best performing on train and test sets. 
  #
  # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the 
  # other two are meant to be off. That said, if you want to include stopword removal
  # or binarization in your best model, write the code accordingl
  # 
  # Hint: Use filterStopWords(words) defined below

#   BEST_MODEL Detect negation words (not, n't and never) and add NOT_ to each word until the next puctuation.
  def addNegationFeatures(self, words):
    neg_words = []
    negation = False
    for word in words:
        if word in ['not', 'never'] or "n't" in word:
          negation = True

#    neg_feature = re.compile("^not$|never|[a-z]n't$")   # regular expression for not, n't and never
#        if re.search(neg_feature, word):
#            negation = True
        if (word not in (',', '.', '?', '!', ';')) & negation:
            word = "NOT_" + word
            negation = False
        if word in (',', '.', '?', '!', ';'):
            negation = False
        neg_words.append(word)
    return neg_words



  def classify(self, words):
    """
      'words' is a list of words to classify. Return 'pos' or 'neg' classification.
    """
    countTotalReviews = self.countPosReviews + self.countNegReviews
    probPos = math.log(self.countPosReviews / countTotalReviews)    # Prior of positive reviews
    probNeg = math.log(self.countNegReviews / countTotalReviews)    # Prior of negative reviews
    
    
    pos_words = set(self.posText)
    neg_words = set(self.negText)
    self.shared_vocab = pos_words.intersection(neg_words)#  [word for word in pos_words if word in neg_words]
    self.train_vocab = set(self.posText).union(set(self.negText))
#    self.word_ratios = {word : self.posText[word] / self.negText[word] for word in self.shared_vocab}
    self.positive_words = [word for word in open('deps/positive-words.txt', 'r', encoding="ISO-8859-1").read().split("\n") if len(word) > 0]
    self.negative_words = [word for word in open('deps/negative-words.txt', 'r', encoding="ISO-8859-1").read().split("\n") if len(word) > 0]

    # first we need to filter all words that don't exist in the train set
#    words = [word for word in words if word in self.train_vocab]
    # then we filter depending on our flags
    words = words if not self.BOOLEAN_NB else list(set(words))
    # set messes up the order, so we need to add the negationfeatures becaues they are order-dependent
    words = words if not self.BEST_MODEL else [word for word in words if word not in self.stopList]
    words = words if not self.BEST_MODEL else self.addNegationFeatures(words)
    words = words if not self.BEST_MODEL else list(set(words))
    if self.BEST_MODEL:
        for word in words:
            if word in self.negative_words or word in self.positive_words:
                words = words + 3 * [word]

    words = words if not self.FILTER_STOP_WORDS else [word for word in words if word not in self.stopList]




    # finally remove punctuation - not necessarily good. Let's see.
    words = [word for word in words if word not in ["," , "." , "-" , "(" , ")" , "-" , "!" , "?" , "[" , "]"]]


    vocab_size = len(list(self.posText.keys())) + len(list(self.negText.keys()))
    pos_factors = [math.log((self.posText.get(word, 0) + 1.0 ) / (self.countPosWords + vocab_size)) for word in words]
    neg_factors = [math.log((self.negText.get(word, 0) + 1.0 ) / (self.countNegWords + vocab_size)) for word in words]
    pos_guess = sum(pos_factors)
    neg_guess = sum(neg_factors) 
#    print("guesses, pos, neg:", pos_guess, neg_guess)
    return 'pos' if pos_guess > neg_guess else 'neg'



  def addExample(self, klass, words):
    """
     * Train your model on an example document with label klass ('pos' or 'neg') and
     * words, a list of strings.
     * You should store whatever data structures you use for your classifier
     * in the NaiveBayes class.
     * Returns nothing
    """
    words = words if not self.FILTER_STOP_WORDS else [word for word in words if word not in self.stopList]
    words = words if not self.BOOLEAN_NB else list(set(words))
     
    
    # set messes up the order, so we need to add the negationfeatures becaues they are order-dependent
    words = words if not self.BEST_MODEL else self.addNegationFeatures(words)
#    words = words if not self.BEST_MODEL else list(set(words))

    # remove puncutation. Could be good or bad. Not sure.
    words = [word for word in words if word not in ["," , "." , "-" , "(" , ")" , "-" , "!" , "?" , "[" , "]"]]



    if klass == 'pos':
      self.countPosReviews += 1
      for word in words:              # use this line for regular Naive Bayes method
      # for word in list(set(words))  # use this line for Boolean Naive Bayes method
        self.posText[word] = self.posText.get(word,0) + 1
        self.countPosWords += 1
        self.text[word] = self.text.get(word,0) + 1
    else:
      self.countNegReviews += 1
      for word in words:              # use this line for regular Naive Bayes method
      # for word in list(set(words))  # use this line for Boolean Naive Bayes method
        self.negText[word] = self.negText.get(word,0) + 1
        self.countNegWords += 1
        self.text[word] = self.text.get(word,0) + 1
    
#    pass #why is there a pass here?

  def filterStopWords(self, words):
    """
    * Filters stop words found in self.stopList.
    """
    filtered_words =[]
    for word in words:
        if word not in self.stopList:
            filtered_words.append(word)
    return filtered_words
    
    
      
  # END TODO (Modify code beyond here with caution)
  #############################################################################
  
  
  def readFile(self, fileName):
    """
     * Code for reading a file.  you probably don't want to modify anything here, 
     * unless you don't like the way we segment files.
    """
    contents = []
    f = open(fileName)
    for line in f:
      contents.append(line)
    f.close()
    result = self.segmentWords('\n'.join(contents)) 
    return result

  
  def segmentWords(self, s):
    """
     * Splits lines on whitespace for file reading
    """
    return s.split()

  
  def trainSplit(self, trainDir):
    """Takes in a trainDir, returns one TrainSplit with train set."""
    split = self.TrainSplit()
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    for fileName in posTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
      example.klass = 'pos'
      split.train.append(example)
    for fileName in negTrainFileNames:
      example = self.Example()
      example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
      example.klass = 'neg'
      split.train.append(example)
    return split

  def train(self, split):
    for example in split.train:
      words = example.words
      if self.FILTER_STOP_WORDS:
        words =  self.filterStopWords(words)
      self.addExample(example.klass, words)
      if self.BEST_MODEL:
        words = self.addNegationFeatures(words)
      self.addExample(example.klass, words)


  def crossValidationSplits(self, trainDir):
    """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
    splits = [] 
    posTrainFileNames = os.listdir('%s/pos/' % trainDir)
    negTrainFileNames = os.listdir('%s/neg/' % trainDir)
    #for fileName in trainFileNames:
    for fold in range(0, self.numFolds):
      split = self.TrainSplit()
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        if fileName[2] == str(fold):
          split.test.append(example)
        else:
          split.train.append(example)
      yield split

  def test(self, split):
    """Returns a list of labels for split.test."""
    labels = []
    for example in split.test:
      words = example.words
      guess = self.classify(words)
      labels.append(guess)
    return labels

  def buildSplits(self, args):
    """Builds the splits for training/testing"""
    trainData = [] 
    testData = []
    splits = []
    trainDir = args[0]
    if len(args) == 1: 
      print('[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir))

      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fold in range(0, self.numFolds):
        split = self.TrainSplit()
        for fileName in posTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
          example.klass = 'pos'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        for fileName in negTrainFileNames:
          example = self.Example()
          example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
          example.klass = 'neg'
          if fileName[2] == str(fold):
            split.test.append(example)
          else:
            split.train.append(example)
        splits.append(split)
    elif len(args) == 2:
      split = self.TrainSplit()
      testDir = args[1]
      print('[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir))
      posTrainFileNames = os.listdir('%s/pos/' % trainDir)
      negTrainFileNames = os.listdir('%s/neg/' % trainDir)
      for fileName in posTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
        example.klass = 'pos'
        split.train.append(example)
      for fileName in negTrainFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
        example.klass = 'neg'
        split.train.append(example)

      posTestFileNames = os.listdir('%s/pos/' % testDir)
      negTestFileNames = os.listdir('%s/neg/' % testDir)
      for fileName in posTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/pos/%s' % (testDir, fileName)) 
        example.klass = 'pos'
        split.test.append(example)
      for fileName in negTestFileNames:
        example = self.Example()
        example.words = self.readFile('%s/neg/%s' % (testDir, fileName)) 
        example.klass = 'neg'
        split.test.append(example)
      splits.append(split)
    return splits

  def filterStopWords(self, words):
    """Filters stop words."""
    filtered = []
    for word in words:
      if not word in self.stopList and word.strip() != '':
        filtered.append(word)
    return filtered


def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
  nb = NaiveBayes()
  splits = nb.buildSplits(args)
  avgAccuracy = 0.0
  fold = 0
  for split in splits:
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    accuracy = 0.0
    # this is where we should filter all the words, before we pass them in to the trainer and classifier
    # and passing them in one at a time is stupid
    for example in split.train:
      words = example.words 
      classifier.addExample(example.klass, words)
    for example in split.test:
      words = example.words
      guess = classifier.classify(words)
      if example.klass == guess:
        accuracy += 1.0

    accuracy = accuracy / len(split.test)
    avgAccuracy += accuracy
    print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy) )
    fold += 1
  avgAccuracy = avgAccuracy / fold
  print('[INFO]\tAccuracy: %f' % avgAccuracy)
    
    
def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
  classifier = NaiveBayes()
  classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
  classifier.BOOLEAN_NB = BOOLEAN_NB
  classifier.BEST_MODEL = BEST_MODEL
  trainSplit = classifier.trainSplit(trainDir)
  classifier.train(trainSplit)
  testFile = classifier.readFile(testFilePath)
  print(classifier.classify(testFile))


def main():
  FILTER_STOP_WORDS = False
  BOOLEAN_NB = False
  BEST_MODEL = False
  (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
  if ('-f','') in options:
    FILTER_STOP_WORDS = True
  elif ('-b','') in options:
    BOOLEAN_NB = True
  elif ('-m','') in options:
    BEST_MODEL = True
  
  if len(args) == 2 and os.path.isfile(args[1]):
    classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
  else:
    test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)

  if __name__ == "__main__":
    main()




