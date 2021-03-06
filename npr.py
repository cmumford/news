#!/usr/bin/env python3

import codecs
import collections
import concurrent.futures
import copy
import csv
import datetime
import dateutil.parser
import doctest
import glob
import itertools
import json
import multiprocessing
import nltk.data
import nltk.classify.util
from nltk import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import nltk.data
from nltk.stem import PorterStemmer
from operator import attrgetter
from pprint import pprint
import re
import signal
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import string
import sys
import threading
import time
import urllib
import xml.etree.cElementTree as ET

# Uses the NPR API: http://api.npr.org/
# Query generator: http://www.npr.org/api/queryGenerator.php

# Last startNum retrieved was 154534

num_cpus = multiprocessing.cpu_count()
keep_running = True

def handler(signum, frame):
  """The signal handling function. Just clears keep_running"""
  print('Signal handler called with signal', signum)
  global keep_running
  if signum == signal.SIGINT:
    keep_running = False

class Topics(object):
  All = 3002
  Columns = 3003
  Programs = 3004
  Series = 3006
  Bios = 3007
  MusicArtists = 3009
  Blogs = 3013
  Tags = 3024

class Tag(object):
  def __init__(self, id, num, title, additionalInfo):
    """Tag constructor"""
    self.id_ = id
    self.num_ = num
    self.title_ = title
    self.additionalInfo_ = additionalInfo

  def __str__(self):
    return 'id:%d, num:%d, "%s"' % (self.id_, self.num_, self.title_)

class Story(object):
  """A single story read from the NPR story corpus."""
  copyright_re = re.compile(r'\[Copyright \d+ NPR\]')
  exclude = set(string.punctuation)
  exclude.add(u'\u2013') # endash
  exclude.add(u'\u2014') # emdash
  excludeStr = ''.join(exclude)

  def __init__(self, id):
    self.id_ = id
    self.title_ = ''
    self.teaser_ = ''
    self.date_ = None
    self.tags_ = set()
    self.text_ = []
    self.url_ = ''

  def text(self):
    """Return all story paragraphs concatenated as a single string."""
    return ' '.join(self.text_)

  def hasText(self):
    """Does this story have any text? Some (like podcasts) do not."""
    return len(self.text_) > 0

  @staticmethod
  def isCopyrightSentence(sentence):
    """Is the string sentence passed in a copyright sentence?"""
    return Story.copyright_re.match(sentence) != None

  @staticmethod
  def stripPunctuation(text):
    """Remove all punctuation from the supplied text string.
    >>> Story.stripPunctuation("Fred")
    'Fred'
    >>> Story.stripPunctuation(r'"Fred"')
    'Fred'
    >>> Story.stripPunctuation(r"Fred's")
    "Fred's"
    >>> Story.stripPunctuation("\\"Fred's\\"")
    "Fred's"
    """
    return text.strip(Story.excludeStr)

  @staticmethod
  def extractWords(sentence):
    """Extract an array of words in a sentence
    >>> Story.extractWords('Hello there man')
    ['Hello', 'there', 'man']
    >>> Story.extractWords('Hello  there man.')
    ['Hello', 'there', 'man']
    """
    return word_tokenize(Story.stripPunctuation(sentence))

  def rawText(self):
    """Strip out punctuation, and return all story text suitable for analysis."""
    words = []
    for para in self.text_:
      newSentence = True
      para = re.sub(Story.copyright_re, '', para)
      for word in para.split():
        word.strip()
        if newSentence:
          if len(word) > 1:
            word = word.lower()
          newSentence = False
        newSentence = word.endswith('.')
        words.append(Story.stripPunctuation(word))
    text = ' '.join(words)
    return re.sub(Story.copyright_re, '', text)

  def switchTags(self):
    """Switch to the "local" tags of the same as the one in this story."""
    self.tags_ = set([NPR.tags[t.id_] for t in self.tags_])

  def hasATag(self, tags):
    """Does the story have at least one of any of |tags|?"""
    for tag in tags:
      if tag in self.tags_:
        return True
    return False

class GenderStats(object):
  def __init__(self, title):
    self.title = title
    self.total = 0
    self.youth = 0
    self.cancer = 0

  @staticmethod
  def totalCounts(counts):
    total = 0
    for tag in counts:
      total += counts[tag]
    return total

  def addTotal(self, counts):
    self.total = GenderStats.totalCounts(counts)

  def addCancer(self, counts):
    self.cancer = GenderStats.totalCounts(counts)

  def addYouth(self, counts):
    self.youth = GenderStats.totalCounts(counts)

  def __str__(self):
    return "%s total:%d cancer:%d youth:%d" % (self.title, self.total, \
                                               self.cancer, self.youth)

  @staticmethod
  def csvHeader():
    return "Sex,Cancer,Youth,Total"

  def asCsv(self):
    return "%s,%d,%d,%d" % (self.title, self.cancer, self.youth, self.total)

class ProgressPrinter(object):
  print_delay = datetime.timedelta(seconds=1)

  def __init__(self, title, meter, max_count):
    self.count = 0
    self.title = title
    self.meter = meter
    self.max_count = max_count
    self.start_time = None
    self.next_print_time = None
    self.lock = threading.Lock()

  def increment(self):
    with self.lock:
      now = datetime.datetime.now()
      if not self.start_time:
        self.start_time = now - ProgressPrinter.print_delay
        self.next_print_time = now
      self.count += 1
      if now >= self.next_print_time:
        elapsed = datetime.datetime.now() - self.start_time
        items_per_sec = self.count / elapsed.total_seconds()
        if self.max_count > 0:
          percent = self.count * 100.0 / self.max_count
          remaining_secs = (self.max_count - self.count) / items_per_sec
          print('%s: %.1f%%, %s:%.2f, remaining:%ds' % \
                (self.title, percent, self.meter, items_per_sec,
                 remaining_secs),
                file=sys.stderr)
        else:
          print('%s: %.d, %s:%.2f' % (self.title, self.count, self.meter,
                                      items_per_sec))
        self.next_print_time = now + self.print_delay

class StoryFileReader(object):
  def __init__(self, apply_auto_tags):
    self.apply_auto_tags = apply_auto_tags

  def __call__(self, file_name):
    return NPR.loadStoriesFromFile(file_name, self.apply_auto_tags)

# Read a collection of story files.
class StoryReader(object):
  sleepSecs = 1

  def __init__(self, file_names, apply_auto_tags=True):
    self.files_to_read = file_names
    self.apply_auto_tags = apply_auto_tags
    self.stories = []
    self.lock = threading.Lock()
    self.t = threading.Thread(target=self.threadReadFunc)
    self.t.start()
    self.thread_exception = None

  def __iter__(self):
    return self

  def threadReadFunc(self):
    global keep_running

    try:
      reader = StoryFileReader(self.apply_auto_tags)
      progress = ProgressPrinter('StoryFileReader', 'files/sec',
                                 len(self.files_to_read))
      with concurrent.futures.ProcessPoolExecutor() as executor:
        for future in concurrent.futures.as_completed(executor.submit(reader, fn) for fn in self.files_to_read):
          progress.increment()
          if future.exception():
            raise future.exception()
          stories = future.result()
          for story in stories:
            story.switchTags()
          NPR.fixupTags(stories)
          with self.lock:
            self.stories.extend(stories)
    except Exception as e:
      self.thread_exception = e
      raise e

  def __next__(self):
    global keep_running
    while True:
      if not keep_running:
        raise StopIteration
      try:
        with self.lock:
          story = self.stories.pop()
          assert story
          return story
      except IndexError:
        if not self.t.isAlive():
          if self.thread_exception:
            raise self.thread_exception
          raise StopIteration
        # Still running, but no new stories, so wait for more to be read.
        time.sleep(StoryReader.sleepSecs)

class GenderOptions(object):
  def __init__(self, groups, all_tags, ignore_tag_ids):
    self.res = {}
    self.tags = {}
    self.all_tags = set()
    self.all_res = {}
    for group in groups:
      self.res[group] = []
      self.tags[group] = set()
      for re_str in groups[group]:
        reg = re.compile(r'\b%s\b' % re_str, re.IGNORECASE)
        self.res[group].append(reg)
        self.all_res[re_str] = reg
        for tag in all_tags:
          if reg.findall(tag.title_) and not tag.id_ in ignore_tag_ids:
            self.tags[group].add(tag)
            self.all_tags.add(tag)

class MaleOptions(GenderOptions):
  def __init__(self, all_tags, ignore_tag_ids):
    super(MaleOptions, self).__init__({
      'adult' : ['mens?', "men's", "man's", "father'?s?", "grandfather'?s?",
                 'grandpa', 'males?', 'masculism', "men's rights",
                 'penis transplant'],
      'youth' : ['sons?', 'boys?'],
      'cancer': ['prostate cancer']}, all_tags, ignore_tag_ids)

class FemaleOptions(GenderOptions):
  def __init__(self, all_tags, ignore_tag_ids):
    super(FemaleOptions, self).__init__({
      'adult' : ['womens?', "women's", "woman's", "mother'?s?",
                 "grandmother'?s?", 'grandma', 'females?', 'feminism',
                 "women's rights?", 'ovarian transplant', 'breast',
                 'breastfeeding', 'breastmilk'],
      'youth' : ['girls?', 'daughters?', '15girls'],
      'cancer': ['breast cancer', 'mastectomy', 'double mastectomy',
                 'Pink ribbon breast cancer awareness']},
      all_tags, ignore_tag_ids)

def loadTags():
  root = ET.parse('tags.xml').getroot()
  tags = []
  for item in root.findall('item'):
    tags.append(Tag(int(item.get('id')),
                    int(item.get('num')),
                    item.find('title').text,
                    item.find('additionalInfo').text))
  return tags

class NamedCounter(object):
  def __init__(self, name):
    self.name = name
    self.count = 0

  def increment(self, amount=1):
    self.count += amount

  def __str__(self):
    return '%s:%d' % (self.name, self.count)

class GenderCounter(object):
  def __init__(self, title):
    self.title = title
    self.female = NamedCounter('female')
    self.male = NamedCounter('male')

  def increment(self, female_amount, male_amount):
    self.female.increment(female_amount)
    self.male.increment(male_amount)

  def add(self, counter):
    self.female.increment(counter.female.count)
    self.male.increment(counter.male.count)

  def __str__(self):
    return '%s %s %s' % (self.title, self.female, self.male)

class GenderSentimentCounter(object):
  def __init__(self):
    self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    self.positive_words = NPR.readWordList('positive-words.txt')
    self.negative_words = NPR.readWordList('negative-words.txt')

  def __call__(self, file_name):
    pos_counts = GenderCounter('positive')
    neg_counts = GenderCounter('negative')
    for story in NPR.loadStoriesFromFile(file_name):
      for para in story.text_:
        for sentence in self.tokenizer.tokenize(para):
          if Story.isCopyrightSentence(sentence):
            continue
          sentence = sentence.lower()
          sentence_words = Story.extractWords(sentence)
          is_female = NPR.matchRegExes(sentence_words,
                                       NPR.female_options.all_res)
          is_male = NPR.matchRegExes(sentence_words,
                                     NPR.male_options.all_res)
          if is_female or is_male:
            for word in sentence_words:
              if word in self.positive_words:
                if is_male:
                  pos_counts.male.count += 1
                if is_female:
                  pos_counts.female.count += 1
              if word in self.negative_words:
                if is_male:
                  neg_counts.male.count += 1
                if is_female:
                  neg_counts.female.count += 1
    return (pos_counts, neg_counts)

class GenderWordCounter(object):
  def __init__(self):
    self.stop = stopwords.words('english')
    self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    pos_words = NPR.readWordList('pos_emotional_words.txt')
    neg_words = NPR.readWordList('neg_emotional_words.txt')
    self.pos_res = {w: re.compile(w) for w in pos_words}
    self.neg_res = {w: re.compile(w) for w in neg_words}

  def __call__(self, file_name):
    pos_counts = GenderCounter('positive')
    neg_counts = GenderCounter('negative')
    for story in NPR.loadStoriesFromFile(file_name):
      for paragraph in story.text_:
        if Story.isCopyrightSentence(paragraph):
          continue
        for sentence in self.tokenizer.tokenize(paragraph):
          sentence = sentence.lower()
          sentence = [w for w in sentence.split() if w not in self.stop]
          pos_count = None
          neg_count = None
          if NPR.matchRegExes(sentence, NPR.female_options.all_res):
            pos_count = NPR.matchRegExes(sentence, self.pos_res, True)
            neg_count = NPR.matchRegExes(sentence, self.neg_res, True)
            pos_counts.female.count += pos_count
            neg_counts.female.count += neg_count
          if NPR.matchRegExes(sentence, NPR.male_options.all_res):
            if pos_count == None:
              pos_count = NPR.matchRegExes(sentence, self.pos_res, True)
              neg_count = NPR.matchRegExes(sentence, self.neg_res, True)
            pos_counts.male.count += pos_count
            neg_counts.male.count += neg_count
    return (pos_counts, neg_counts)

class FileSentimentAnalyzer(object):
  def __init__(self):
    self.classifier = NPR.createSentimentClassifier()

  def __call__(self, file_name):
    pos_counter = GenderCounter('positive')
    neg_counter = GenderCounter('negative')
    tokenizer = None

    for story in NPR.loadStoriesFromFile(file_name):
      for paragraph in story.text_:
        if Story.isCopyrightSentence(paragraph):
          continue
        if not tokenizer:
          tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        for sentence in tokenizer.tokenize(paragraph):
          sentence_words = Story.extractWords(sentence.lower())
          is_female = NPR.matchRegExes(sentence_words, NPR.female_options.all_res)
          is_male = NPR.matchRegExes(sentence_words, NPR.male_options.all_res)
          if is_female or is_male:
            cl = self.classifier.classify(NPR.extract_features(sentence_words))
            if cl == 'pos':
              pos_counter.increment(int(is_female), int(is_male))
            if cl == 'neg':
              neg_counter.increment(int(is_female), int(is_male))
    return (pos_counter, neg_counter)

class ReadGenderStories(object):
  def __call__(self, file_name):
    stories = []
    for story in NPR.loadStoriesFromFile(file_name):
      for paragraph in story.text_:
        if Story.isCopyrightSentence(paragraph):
          continue
      if story.hasATag(NPR.gender_tags):
        stories.append(story)
    return stories

class NPR(object):
  baseUrl = 'http://api.npr.org/query?'
  ignore_tag_ids = [
    126927651,  # "Mother Jones"
    184560888,  # "Mother's Day Shooting"
    126826632,  # "Mad Men"
    129251919,  # "No Country For Old Men"
    152027155,  # "Beastie Boys"
    131877737,  # "The Blue Rhythm Boys"
  ]
  sports_tag_ids = [
    149849695,  # "NCAA men basketball"
    149849693,  # "NCAA men's basketball"
    135170830   # "NCAA women's basketball"
  ]
  all_tags = loadTags()
  tags = {tag.id_:tag for tag in all_tags}
  tag_titles = {tag.title_:tag for tag in all_tags}
  female_options = FemaleOptions(all_tags, ignore_tag_ids)
  male_options = MaleOptions(all_tags, ignore_tag_ids)
  gender_tags = set()
  gender_tags |= female_options.all_tags
  gender_tags |= male_options.all_tags
  classifier_story_min_count = 17
  classified_stories = None
  stemmer = PorterStemmer()

  def __init__(self, api_key):
    self.api_key_ = api_key

  def getUrl(self, params = {}):
    common_params = {'apiKey': self.api_key_}
    params.update(common_params)
    return NPR.baseUrl + urllib.parse.urlencode(params)

  def downloadData(self):
    params = {'startNum':154534, 'numResults':20}
    story_count = 1    # Any non-zero number to start
    total_stories = 0
    while story_count:
      url = self.getUrl(params)
      print(url)
      f = urllib.urlopen(url)
      xml_response = f.read()
      with open('stories/startNum_%d.xml' % params['startNum'], 'w') as f:
        f.write(xml_response)
      root = ET.fromstring(xml_response)
      story_count = len(root.findall('list/story'))
      total_stories += story_count
      print('there are', story_count, 'stories. So far:', total_stories)
      params['startNum'] = params['startNum'] + story_count

  @staticmethod
  def createSentimentClassifier():
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(NPR.word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(NPR.word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = int(len(negfeats)*3/4)
    poscutoff = int(len(posfeats)*3/4)

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    print('Training on', len(trainfeats), 'instances...')

    classifier = NaiveBayesClassifier.train(trainfeats)
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    print('Testing on', len(testfeats), 'instances, accuracy:',
           nltk.classify.util.accuracy(classifier, testfeats))
    classifier.show_most_informative_features()
    return classifier

  @staticmethod
  def loadStoriesFromFile(file_name, apply_missing_tags=True):
    auto_tagged_stories = None
    if apply_missing_tags:
      if not NPR.classified_stories:
        NPR.classified_stories = NPR.loadStoriesFromFile('classified_stories.xml', False)
      auto_tagged_stories = {s.id_:s for s in NPR.classified_stories}
    stories = []
    root = ET.parse(file_name).getroot()
    for xml_story in root.findall('list/story'):
      story = Story(int(xml_story.get('id')))
      story.title_ = xml_story.find('title').text
      links = xml_story.findall("link[@type='short']")
      if len(links) == 1:
        story.url_ = links[0].text
      if not story.url_:
        links = xml_story.findall("link[@type='html']")
        if len(links) == 1:
          story.url_ = links[0].text

      xml_teaser = xml_story.find('teaser')
      if xml_teaser:
        story.teaser_ = xml_teaser.text
      story.date_ = dateutil.parser.parse(xml_story.find('storyDate').text)
      stories.append(story)
      for parent in xml_story.findall("parent[@type='tag']"):
        tag_id = int(parent.get('id'))
        tag = NPR.tags[tag_id]
        story.tags_.add(tag)
      # If this story was auto-tagged then apply those as well.
      if auto_tagged_stories and story.id_ in auto_tagged_stories:
        story.tags_ |= auto_tagged_stories[story.id_].tags_
      for text in xml_story.findall("text/paragraph"):
        if text.text:
          story.text_.append(text.text)
    return stories

  def writeStoriesToXml(self, stories, fname):
    root = ET.Element("nprml")
    xml_list = ET.SubElement(root, "list")
    for story in stories:
      xml_story = ET.SubElement(xml_list, "story", id=str(story.id_))
      ET.SubElement(xml_story, "title").text = story.title_
      ET.SubElement(xml_story, "storyDate").text = str(story.date_)
      if story.teaser_:
        ET.SubElement(xml_story, "teaser").text = story.teaser_
      if story.url_:
        # This isn't technically correct because |url_| could be from either
        # the short link, or the long one.
        ET.SubElement(xml_story, "link", type='short').text = story.url_
      for tag in story.tags_:
        parent = ET.SubElement(xml_story, "parent", type='tag', id=str(tag.id_))
        ET.SubElement(parent, "title").text = tag.title_
      para_idx = 1
      for text in story.text_:
        xml_text = ET.SubElement(xml_story, "text")
        t = text
        ET.SubElement(xml_text, "paragraph", num=str(para_idx)).text = t
        para_idx += 1
    tree = ET.ElementTree(root)
    tree.write(fname)

  @staticmethod
  def one_in_another(container_a, container_b):
    for c in container_a:
      if c in container_b:
        return True
    return False

  @staticmethod
  def one_not_in_another(container_a, container_b):
    for c in container_a:
      if c not in container_b:
        return True
    return False

  @staticmethod
  def matchRegExes(sentence, regexes, count_all=False):
    count = 0
    if type(sentence).__name__ == 'str':
      for reg in regexes:
        if regexes[reg].findall(sentence):
          count += 1
          if not count_all:
            break
    else:
      for word in sentence:
        for reg in regexes:
          if regexes[reg].match(word):
            count += 1
            if not count_all:
              break
    return count

  @staticmethod
  def calcTagCounts(stories, tags):
    totals = {}
    for tag in tags:
      totals[tag] = 0
    for story in stories:
      for tag in story.tags_:
        if tag in totals:
          totals[tag] = totals[tag] + 1
    return totals

  @staticmethod
  def printDictAsCSV(d, fname):
    the_dict = {tag.title_: d[tag] for tag in d}
    with open(fname, 'w') as f:
      w = csv.DictWriter(f, the_dict)
      w.writeheader()
      w.writerow(the_dict)

  @staticmethod
  def printTags(title, stories, tags, min_count = 0):
    print('%s tags' % title)
    print('=========')
    counts = {}
    for story in stories:
      for tag in story.tags_:
        if tag in counts:
          counts[tag] = counts[tag] + 1
        else:
          counts[tag] = 1
    total = 0
    get_key = attrgetter('title_')
    for tag in sorted(tags, key = lambda mbr: get_key(mbr).lower()):
      count = counts[tag] if tag in counts else 0
      if count >= min_count:
        total += count
        print('%s:%d' % (tag.title_, count))
    print('---------')
    print('Total:%d' % total)

  @staticmethod
  def printAllTags(stories, min_count = 0):
    print()
    NPR.printTags('Female', stories, NPR.female_options.all_tags, min_count)
    print()
    NPR.printTags('Male', stories, NPR.male_options.all_tags, min_count)

  def analyzeGenderTitles(self):
    file_names = glob.glob('stories/*.xml')
    progress = ProgressPrinter('Titles', 'files/sec', len(file_names))
    matcher = ReadGenderStories()
    counts = {
      2011: GenderCounter('2011'),
      2012: GenderCounter('2012'),
      2013: GenderCounter('2013'),
      2014: GenderCounter('2014'),
      2015: GenderCounter('2015'),
    }
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
      futures = [executor.submit(matcher, fn) for fn in file_names]
      for future in concurrent.futures.as_completed(futures):
        progress.increment()
        if future.exception() is not None:
          raise future.exception()
        for story in future.result():
          if NPR.matchRegExes(story.title_, NPR.female_options.all_res):
            counts[story.date_.year].female.increment()
          if NPR.matchRegExes(story.title_, NPR.male_options.all_res):
            counts[story.date_.year].male.increment()

    print(GenderStats.csvHeader())
    for year in range(2011, 2016):
      print(counts[year])

  def analyzeGenderStories(self):
    file_names = glob.glob('stories/*.xml')
    progress = ProgressPrinter('Gender Counter', 'files/sec', len(file_names))
    matching_stories = []
    matcher = ReadGenderStories()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
      futures = [executor.submit(matcher, fn) for fn in file_names]
      for future in concurrent.futures.as_completed(futures):
        progress.increment()
        if future.exception() is not None:
          raise future.exception()
        stories = future.result()
        for story in stories:
          story.switchTags()
        matching_stories.extend(stories)
    print('Have:', len(matching_stories), 'gender stories')

    print(GenderStats.csvHeader())
    for year in range(2011, 2016):
      stories = []
      for story in matching_stories:
        if story.date_.year == year:
          stories.append(story)

      male = GenderStats('Male')
      female = GenderStats('Female')

      counts = NPR.calcTagCounts(stories, NPR.female_options.all_tags)
      NPR.printDictAsCSV(counts, 'analysis_female.csv')
      female.addTotal(counts)

      counts = NPR.calcTagCounts(stories, NPR.female_options.tags['cancer'])
      NPR.printDictAsCSV(counts, 'analysis_female_cancer.csv')
      female.addCancer(counts)

      counts = NPR.calcTagCounts(stories, NPR.female_options.tags['youth'])
      NPR.printDictAsCSV(counts, 'analysis_girls.csv')
      female.addYouth(counts)

      counts = NPR.calcTagCounts(stories, NPR.male_options.all_tags)
      NPR.printDictAsCSV(counts, 'analysis_male.csv')
      male.addTotal(counts)

      counts = NPR.calcTagCounts(stories, NPR.male_options.tags['cancer'])
      NPR.printDictAsCSV(counts, 'analysis_male_cancer.csv')
      male.addCancer(counts)

      counts = NPR.calcTagCounts(stories, NPR.male_options.tags['youth'])
      NPR.printDictAsCSV(counts, 'analysis_boys.csv')
      male.addYouth(counts)

      print(year)
      print(female.asCsv())
      print(male.asCsv())

  def countGenders(self):
    female_counts = {}
    for key in self.female_options.all_res:
      female_counts[key] = 0
    male_counts = {}
    for key in self.male_options.all_res:
      male_counts[key] = 0

    for story in StoryReader(glob.glob('stories/*.xml')):
      story_text = story.rawText()
      for reg in self.female_options.all_res:
        female_counts[reg] += len(self.female_options.all_res[reg].findall(story_text))
      for reg in self.male_options.all_res:
        male_counts[reg] += len(self.male_options.all_res[reg].findall(story_text))

    print('Female counts:')
    print('==============')
    total = 0
    for reg in female_counts:
      print('%s: %d' % (reg, female_counts[reg]))
      total += female_counts[reg]
    print('--------------')
    print('Total:', total)

    print()
    print('Male counts:')
    print('============')
    total = 0
    for reg in male_counts:
      print('%s: %d' % (reg, male_counts[reg]))
      total += male_counts[reg]
    print('--------------')
    print('Total:', total)

  @staticmethod
  def readWordList(fname):
    words = set()
    with codecs.open(fname, 'r', 'iso-8859-1') as f:
      for line in f.readlines():
        line = line.strip()
        if not line.startswith('#'):
          words.add(line)
    return words

  def countGenderSentiments(self):
    counter = GenderSentimentCounter()
    pos = GenderCounter('positive')
    neg = GenderCounter('negative')
    file_names = glob.glob('stories/*.xml')
    progress = ProgressPrinter('SentimentAnalyzer', 'files/sec',
                               len(file_names))
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
      futures = [executor.submit(counter, fn) for fn in file_names]
      for future in concurrent.futures.as_completed(futures):
        progress.increment()
        if future.exception() is not None:
          raise future.exception()
        (p, n) = future.result()
        pos.add(p)
        neg.add(n)
    print(pos)
    print(neg)

  @staticmethod
  def word_feats(words):
    return dict([(word, True) for word in words])

  @staticmethod
  def extract_features(document):
    document_words = set(document)
    features = {}
    for word in movie_reviews.words():
      features['contains(%s)' % word] = (word in document_words)
    return features

  def analyzeSentiments(self):
    analyzer = FileSentimentAnalyzer()
    pos_counter = GenderCounter('positive')
    neg_counter = GenderCounter('negative')
    file_names = glob.glob('stories/*.xml')
    progress = ProgressPrinter('FileSentimentAnalyzer', 'files/sec',
                               len(file_names))
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_cpus*3/2)) as executor:
      for future in concurrent.futures.as_completed(executor.submit(analyzer, fn) for fn in file_names):
        progress.increment()
        if future.exception() is not None:
          raise future.exception()
        (pos, neg) = future.result()
        pos_counter.add(pos)
        neg_counter.add(neg)
    print('%s, %s' %(pos_counter, neg_counter))

  def analyzeWords(self):
    counter = GenderWordCounter()
    pos = GenderCounter('positive')
    neg = GenderCounter('negative')
    file_names = glob.glob('stories/*.xml')
    progress = ProgressPrinter('analyzeWords', 'files/sec', len(file_names))
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(num_cpus*3/2)) as executor:
      for future in concurrent.futures.as_completed(executor.submit(counter, fn) for fn in file_names):
        progress.increment()
        if future.exception() is not None:
          raise future.exception()
        else:
          (p, n) = future.result()
          pos.add(p)
          neg.add(n)
    print(pos)
    print(neg)

  @staticmethod
  def readWordList(fname):
    words = set()
    with codecs.open(fname, encoding='latin-1') as f:
      for line in f:
        items = line.split()
        if len(items) and not items[0].startswith('#'):
          words.add(items[0])
    return words

  def countAttribute(self, attr_name):
    counter = GenderCounter(attr_name)
    for story in StoryReader(glob.glob('stories/*.xml')):
      attr = getattr(story, attr_name)
      if not attr:
        continue
      attr = attr.lower()
      male_count = 0
      female_count = 0
      for reg in NPR.male_options.all_res.values():
        male_count += len(reg.findall(attr))
      for reg in NPR.female_options.all_res.values():
        female_count += len(reg.findall(attr))
      counter.increment(female_count, male_count)
    print(counter)

  @staticmethod
  def textTokenizer(text, stem=True):
    """ Tokenize text and stem words removing punctuation """
    text = Story.stripPunctuation(text)
    tokens = word_tokenize(text)
    if stem:
      tokens = [NPR.stemmer.stem(t) for t in tokens]

    return tokens

  @staticmethod
  def newClusteringPipeline(stories, num_clusters=20):
    texts = []
    classify_stories = []
    progress = ProgressPrinter('Clustering', 'stories/sec', 0)
    count = 0
    for story in stories:
      if not story.hasText():
        continue
      count += 1
      if count > 36000:
        break
      t = story.text()
      assert t
      texts.append(t)
      classify_stories.append(story)
    print('There are', len(classify_stories), 'stories to classify')
    assert len(classify_stories)

    vectorizer = TfidfVectorizer(tokenizer=NPR.textTokenizer,
                                 stop_words=stopwords.words('english'),
                                 max_df=0.5,
                                 min_df=0.1,
                                 lowercase=True)

    print('fitting to text', file=sys.stderr)
    tfidf_model = vectorizer.fit_transform(texts)
    km_model = KMeans(n_clusters=num_clusters, n_jobs=-1)
    km_model.fit(tfidf_model)

    clustering = collections.defaultdict(list)

    for idx, label in enumerate(km_model.labels_):
      story = classify_stories[idx]
      clustering[label].append((story.title_, ', '.join([t.title_ for t in story.tags_])))

    print('Done clustering')
    return clustering

  @staticmethod
  def newClassifierPipeline(stories, min_tag_count = 1,
                            always_include_tags = None):
    stories_with_text = []
    tag_counts = {}
    # First scan to calculate counts
    for story in stories:
      if not story.hasText():
        continue
      stories_with_text.append(story)
      for tag in story.tags_:
        if tag in tag_counts:
          tag_counts[tag] += 1
        else:
          tag_counts[tag] = 1

    # Now gather the data for stories with enough tags.
    tags = []
    data = []
    targets = []
    progress = ProgressPrinter('Classifier', 'stories/sec', len(stories_with_text))
    for story in stories_with_text:
      progress.increment()
      tt = []
      for tag in story.tags_:
        if tag_counts[tag] >= min_tag_count or \
                (always_include_tags and tag in always_include_tags):
          if tag not in tags:
            tags.append(tag)
          tt.append(tag.title_)
      if len(tt):
        data.append(story.rawText())
        targets.append(tt)

    print('Will be classifying using the following existing tags:')
    NPR.printTags('Classification Tags', stories_with_text, tags)

    print("Transforming targets...")
    lb = MultiLabelBinarizer()
    Y = lb.fit_transform(targets)

    print("Training classifier...", file=sys.stderr)
    text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),
                         ('tfidf', TfidfTransformer()),
                         ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=-1))
                        ])
    return (text_clf.fit(data, Y), lb)

  @staticmethod
  def createClassifierPipeline(min_tag_count=1, always_include_tags=None,
                               apply_auto_tags=False):
    return NPR.newClassifierPipeline(StoryReader(glob.glob('stories/*.xml'),
                                                 apply_auto_tags),
                                     min_tag_count,
                                     always_include_tags)

  def classifyStories(self):
    (cl, lb) = NPR.createClassifierPipeline(NPR.classifier_story_min_count,
                                            NPR.gender_tags)

    # Now fit *all* stories
    story_texts = []
    for story in stories:
      story_texts.append(story.rawText())

    print("Predicting stories...")
    predicted = cl.predict(story_texts)
    print("Printing results...")
    all_labels = lb.inverse_transform(predicted)
    adjusted_tags = []
    with open('categorized.txt', 'w') as f:
      for story, categories in zip(stories, all_labels):
        print('%s => %s' % (story.title_, ','.join(categories) ), file=f)
        if len(categories):
          cl_tags = set([NPR.tag_titles[c] for c in categories])
          if NPR.one_not_in_another(cl_tags, story.tags_):
            if False:
              print('Title:', story.title_)
              print('  url:', story.url_)
              print('  npr:', ', '.join([t.title_ for t in story.tags_]))
              print('  CLS:', ', '.join(categories))
              print()
            story.tags_ = story.tags_
            story.tags_ |= cl_tags
            adjusted_tags.append(story)

    self.writeStoriesToXml(adjusted_tags, 'classified_stories.xml')

  @staticmethod
  def fixupTags(stories):
    # NPR's tags are fairly messy. This does some very basic
    fixes = {
      # Female
      "#15Girls": ["15girls"],
      "black women and marriage": ["black women", "Marriage"],
      "breast cacn": ["breast cacn"],
      "breast cancer, mastectomy, Samantha Harris": ["breast cancer", "mastectomy"],
      "daughter": ["daughters"],
      "girl": ["Girls"],
      "teenage girls": ["teen girls"],
      "women in tech, data": ["women in tech"],
      "women's right": ["women's rights"],
      "women's world cup soccer": ["women's World Cup"],
      "breast-feeding": ["breastfeeding"],
      # Male
      "Boy Scouts": ["Boy Scouts of America"],
      "NCAA men basketball": ["NCAA men's basketball"],
    }

    tags = {}
    for key in fixes:
      tags[NPR.tag_titles[key]] = [NPR.tag_titles[t] for t in fixes[key]]

    for story in stories:
      new_tags = set()
      for tag in story.tags_:
        if tag in tags:
          new_tags |= set(tags[tag])
        else:
          new_tags.add(tag)
      story_tags_ = new_tags

if __name__ == '__main__':
  doctest.testmod()
  try:
    api_key = open('key.txt').read().strip()
    npr = NPR(api_key)
    npr.analyzeGenderStories()
  except:
    keep_running = False
    raise
