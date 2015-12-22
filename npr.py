#!/usr/bin/env python

import copy
import csv
import datetime
import dateutil.parser
import glob
import itertools
import json
from operator import attrgetter
import re
import signal
import string
import sys
import thread
import threading
import time
import urllib
import xml.etree.cElementTree as ET

# Uses the NPR API: http://api.npr.org/
# Query generator: http://www.npr.org/api/queryGenerator.php

# Last startNum retrieved was 154534

keep_running = True

def handler(signum, frame):
  print 'Signal handler called with signal %s' % signum
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
    self.id_ = id
    self.num_ = num
    self.title_ = title
    self.additionalInfo_ = additionalInfo

  def __unicode__(self):
    return 'id:%d, num:%d, "%s"' % (self.id_, self.num_, self.title_)

  def __str__(self):
    return unicode(self).encode('utf-8')

class Story(object):
  copyright_re = re.compile(r'Copyright \d+ NPR')
  exclude = set(string.punctuation)
  exclude.add(u'\u2013') # endash
  exclude.add(u'\u2014') # emdash
  excludeStr = ''.join(exclude)

  def __init__(self, id):
    self.id_ = id
    self.title_ = ''
    self.date_ = None
    self.tags_ = []
    self.text_ = []

  def text(self):
    return ' '.join(self.text_)

  def hasText(self):
    return len(self.text_) > 0

  @staticmethod
  def stripPunctuation(word):
    return word.strip(Story.excludeStr)

  # Strip out punctuation, and return all story text suitable for analysis.
  def rawText(self):
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

  def hasATag(self, tags):
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

  def asCsv(self):
    return "Sex,Total,Cancer,Youth\n%s,%d,%d,%d" % \
        (self.title, self.total, self.cancer, self.youth)

# Read a collection of story files.
class StoryReader(object):
  sleepSecs = 1

  def __init__(self, npr, file_names):
    self.npr = npr
    self.files_to_read = file_names
    self.stories = []
    self.lock = thread.allocate_lock()
    self.t = threading.Thread(target=self.threadReadFunc)
    self.t.start()

  def __iter__(self):
    return self

  def threadReadFunc(self):
    global keep_running
    num_files_read = 0
    start_time = datetime.datetime.now()
    print_delay = datetime.timedelta(seconds=1)
    next_print_time = start_time
    num_files = len(self.files_to_read)
    while keep_running:
      file_name = None
      try:
        self.lock.acquire()
        file_name = self.files_to_read.pop()
      except IndexError:
        pass
      finally:
        self.lock.release()
      if not file_name:
        break
      stories = self.npr.loadStoriesFromFile(file_name)

      num_files_read += 1
      now = datetime.datetime.now()
      if now >= next_print_time:
        elapsed = datetime.datetime.now() - start_time
        files_per_sec = num_files_read / elapsed.total_seconds()
        percent = num_files_read * 100.0 / num_files
        remaining_secs = (num_files - num_files_read) / files_per_sec
        print >> sys.stderr,  '%s: %.1f%%, fps:%.1f, remaining:%ds' % \
            (file_name, percent, files_per_sec, remaining_secs)
        next_print_time = now + print_delay

      try:
        self.lock.acquire()
        self.stories.extend(stories)
      finally:
        self.lock.release()

  def next(self):
    try:
      self.lock.acquire()
      num_files_left = len(self.files_to_read)
      story = self.stories.pop()
    except IndexError:
      story = None
    finally:
      self.lock.release()

    if story:
      return story

    if not num_files_left:
      raise StopIteration

    # If here then we're iterating faster than stories can be read, so wait for
    # more stories to be added.
    while not story:
      time.sleep(StoryReader.sleepSecs)
      try:
        self.lock.acquire()
        story = self.stories.pop()
      except IndexError:
        pass
      finally:
        self.lock.release()
    return story

class GenderOptions(object):
  def __init__(self, groups):
    self.res = {}
    self.all_res = {}
    for group in groups:
      res = []
      for re_str in groups[group]:
        reg = re.compile(r'\b%s\b' % re_str, re.IGNORECASE)
        res.append(reg)
        self.all_res[re_str] = reg
      self.res[group] = res

class MaleOptions(GenderOptions):
  def __init__(self):
    super(MaleOptions, self).__init__({
      'adult' : ['mens?', "men's", "man's", "father'?s?", "grandfather'?s?",
                 'grandpa', 'males?', 'masculism', "men's rights"],
      'youth' : ['sons?', 'boys?', 'grandpa'],
      'cancer': ['prostate cancer']})

class FemaleOptions(GenderOptions):
  def __init__(self):
    super(FemaleOptions, self).__init__({
      'adult' : ['womens?', "women's", "woman's", "mother'?s?",
                 "grandmother'?s?", 'grandma', 'females?', 'feminism',
                 "women's rights?", 'ovarian transplant'],
      'youth' : ['girls?', 'daughters?', '15girls'],
      'cancer': ['breast cancer']})

class NPR(object):
  baseUrl = 'http://api.npr.org/query?'
  all_tags = []
  tags = {}
  girl_tags = set()
  female_tags = set()
  boy_tags = set()
  male_tags = set()
  female_cancer_tags = set()
  male_cancer_tags = set()
  female_stories = set()
  male_stories = set()
  female_options = FemaleOptions()
  male_options = MaleOptions()

  def __init__(self, api_key):
    NPR.loadTagsOfInterest()
    self.api_key_ = api_key

  @staticmethod
  def loadTags():
    root = ET.parse('tags.xml').getroot()
    tags = []
    for item in root.findall('item'):
      tags.append(Tag(int(item.get('id')),
                      int(item.get('num')),
                      item.find('title').text,
                      item.find('additionalInfo').text))
    return tags

  @staticmethod
  def findMatchingTags(reg_str, all_tags):
    tags = set()
    reg = re.compile(reg_str, re.IGNORECASE)
    for tag in all_tags:
      if reg.match(tag.title_):
        tags.add(tag)
    return tags

  @staticmethod
  def findWomenCancerTags(all_tags):
    return NPR.findMatchingTags(r'.*breast cancer.*', all_tags)

  @staticmethod
  def findWomenTags(all_tags):
    tags = set()
    ignore_ids = [
      126927651, # "Mother Jones"
      184560888  # "Mother's Day Shooting"
    ]
    words = ['womens?', 'mothers?', 'girls?', 'daughters?', 'grandmothers?',
             'grandma', 'females?', 'feminism', '#15Girls', '15girls',
             "women's rights?", 'ovarian transplant']
    # Questionable tags. Assuming mostly about women
    words.extend(['sexism'])
    for word in words:
      reg = re.compile(r'.*\b%s\b.*' % word, re.IGNORECASE)
      for tag in all_tags:
        if tag.id_ not in ignore_ids \
            and not NPR.isSportsTag(tag) \
            and reg.match(tag.title_):
          tags.add(tag)
    return tags

  @staticmethod
  def findGirlTags(all_tags):
    tags = set()
    ignore_ids = [
      126927651, # "Mother Jones"
      184560888  # "Mother's Day Shooting"
    ]
    words = ['girls?', 'daughters?', '#15Girls', '15girls']
    # Questionable tags. Assuming mostly about women
    words.extend(['sexism'])
    for word in words:
      reg = re.compile(r'.*\b%s\b.*' % word, re.IGNORECASE)
      for tag in all_tags:
        if tag.id_ not in ignore_ids \
            and not NPR.isSportsTag(tag) \
            and reg.match(tag.title_):
          tags.add(tag)
    return tags

  @staticmethod
  def isSportsTag(tag):
    return tag.id_ in [
      149849695,  # "NCAA men basketball"
      149849693,  # "NCAA men's basketball"
      135170830   # "NCAA women's basketball"
    ]

  @staticmethod
  def findMenTags(all_tags):
    tags = set()
    ignore_ids = [
      126826632,  # "Mad Men"
      129251919,  # "No Country For Old Men"
      152027155,  # "Beastie Boys"
      131877737   # "The Blue Rhythm Boys"
    ]
    words = ['men', "men's", 'fathers?', 'boys?', 'sons?', 'grandfathers?',
             'grandpa', 'male?']
    for word in words:
      reg = re.compile(r'.*\b%s\b.*' % word, re.IGNORECASE)
      for tag in all_tags:
        if tag.id_ not in ignore_ids \
          and not NPR.isSportsTag(tag) \
          and reg.match(tag.title_):
          tags.add(tag)
    return tags

  @staticmethod
  def findBoyTags(all_tags):
    tags = set()
    ignore_ids = [
      126826632,  # "Mad Men"
      129251919,  # "No Country For Old Men"
      152027155,  # "Beastie Boys"
      131877737   # "The Blue Rhythm Boys"
    ]
    words = ['boys?', 'sons?']
    for word in words:
      reg = re.compile(r'.*\b%s\b.*' % word, re.IGNORECASE)
      for tag in all_tags:
        if tag.id_ not in ignore_ids \
          and not NPR.isSportsTag(tag) \
          and reg.match(tag.title_):
          tags.add(tag)
    return tags

  @staticmethod
  def findMaleCancerTags(all_tags):
    return NPR.findMatchingTags(r'.*prostate.*', all_tags)

  def getUrl(self, params = {}):
    common_params = {'apiKey': self.api_key_}
    params.update(common_params)
    return NPR.baseUrl + urllib.urlencode(params)

  @staticmethod
  def getYMD(dt):
    return dt.strftime('%Y-%m-%d')

  def downloadData(self):
    params = {'startNum':154534, 'numResults':20}
    story_count = 1    # Any non-zero number to start
    total_stories = 0
    while story_count:
      url = self.getUrl(params)
      print url
      f = urllib.urlopen(url)
      xml_response = f.read()
      with open('stories/startNum_%d.xml' % params['startNum'], 'w') as f:
        f.write(xml_response)
      root = ET.fromstring(xml_response)
      story_count = len(root.findall('list/story'))
      total_stories += story_count
      print 'there are', story_count, 'stories. So far:', total_stories
      params['startNum'] = params['startNum'] + story_count

  def loadStoriesFromFile(self, file_name):
    stories = []
    root = ET.parse(file_name).getroot()
    for xml_story in root.findall('list/story'):
      story = Story(int(xml_story.get('id')))
      story.title_ = xml_story.find('title').text
      story.date_ = dateutil.parser.parse(xml_story.find('storyDate').text)
      stories.append(story)
      for parent in xml_story.findall("parent[@type='tag']"):
        tag_id = int(parent.get('id'))
        tag = NPR.tags[tag_id]
        story.tags_.append(tag)
      for text in xml_story.findall("text/paragraph"):
        if text.text:
          story.text_.append(text.text)
    return stories

  def writeStoriesToXml(self, stories):
    root = ET.Element("nprml")
    xml_list = ET.SubElement(root, "list")
    for story in stories:
      xml_story = ET.SubElement(xml_list, "story", id=str(story.id_))
      ET.SubElement(xml_story, "title").text = story.title_
      ET.SubElement(xml_story, "storyDate").text = str(story.date_)
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
    tree.write("matching.xml")

  @staticmethod
  def loadTagsOfInterest():
    NPR.all_tags = NPR.loadTags()
    for tag in NPR.all_tags:
      NPR.tags[tag.id_] = tag
    NPR.female_tags = NPR.findWomenTags(NPR.all_tags)
    NPR.female_cancer_tags = NPR.findWomenCancerTags(NPR.all_tags)
    NPR.male_tags = NPR.findMenTags(NPR.all_tags)
    NPR.male_cancer_tags = NPR.findMaleCancerTags(NPR.all_tags)
    NPR.boy_tags = NPR.findBoyTags(NPR.all_tags)
    NPR.girl_tags = NPR.findGirlTags(NPR.all_tags)

    NPR.female_all_tags = set()
    NPR.female_all_tags |= NPR.female_tags
    NPR.female_all_tags |= NPR.female_cancer_tags
    NPR.female_all_tags |= NPR.girl_tags

    NPR.male_all_tags = set()
    NPR.male_all_tags |= NPR.male_tags
    NPR.male_all_tags |= NPR.male_cancer_tags
    NPR.male_all_tags |= NPR.boy_tags

  # Extract a subset of the stories, and write them to a single file for
  # analysis.
  def extractMatchingStories(self):
    combined_tags = copy.copy(NPR.female_all_tags)
    combined_tags |= NPR.male_all_tags

    matching_stories = []
    for story in StoryReader(self, glob.glob('stories/*.xml')):
      if story.hasATag(combined_tags):
        matching_stories.append(story)

    print 'There are', len(matching_stories), 'matching stories'
    npr.writeStoriesToXml(matching_stories)

  @staticmethod
  def one_in_another(container_a, container_b):
    for c in container_a:
      if c in container_b:
        return True
    return False

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
    with open(fname, 'wb') as f:
      w = csv.DictWriter(f, the_dict)
      w.writeheader()
      w.writerow(the_dict)

  def printTags(self, title, stories, tags):
    print '%s tags' % title
    print '========='
    counts = {}
    for story in stories:
      for tag in story.tags_:
        if tag in counts:
          counts[tag] = counts[tag] + 1
        else:
          counts[tag] = 1
    total = 0
    for tag in sorted(tags, key=attrgetter('title_')):
      count = counts[tag] if tag in counts else 0
      total += count
      print '%s:%d' % (tag.title_, count)
    print '---------'
    print 'Total:%d' % total

  def printAllTags(self, stories):
    self.printags('Female', stories, NPR.female_all_tags)
    print
    self.printags('Male', stories, NPR.male_all_tags)

  def analyzeMatchingStories(self):
    matching_stories = self.loadStoriesFromFile('matching.xml')
    print 'Analyzing', len(matching_stories), 'matching stories'

    for year in range(2010, 2016):
      stories = []
      for story in matching_stories:
        if story.date_.year == year:
          stories.append(story)

      male = GenderStats('Male')
      female = GenderStats('Female')

      counts = NPR.calcTagCounts(stories, NPR.female_all_tags)
      NPR.printDictAsCSV(counts, 'analysis_female.csv')
      female.addTotal(counts)

      counts = NPR.calcTagCounts(stories, NPR.female_cancer_tags)
      NPR.printDictAsCSV(counts, 'analysis_female_cancer.csv')
      female.addCancer(counts)

      counts = NPR.calcTagCounts(stories, NPR.girl_tags)
      NPR.printDictAsCSV(counts, 'analysis_girls.csv')
      female.addYouth(counts)

      counts = NPR.calcTagCounts(stories, NPR.male_all_tags)
      NPR.printDictAsCSV(counts, 'analysis_male.csv')
      male.addTotal(counts)

      counts = NPR.calcTagCounts(stories, NPR.male_cancer_tags)
      NPR.printDictAsCSV(counts, 'analysis_male_cancer.csv')
      male.addCancer(counts)

      counts = NPR.calcTagCounts(stories, NPR.boy_tags)
      NPR.printDictAsCSV(counts, 'analysis_boys.csv')
      male.addYouth(counts)

      print year
      print female.asCsv()
      print male.asCsv()

  def countGenders(self):
    female_counts = {}
    for key in self.female_options.all_res:
      female_counts[key] = 0
    male_counts = {}
    for key in self.male_options.all_res:
      male_counts[key] = 0

    for story in StoryReader(self, glob.glob('stories/*.xml')):
      story_text = story.rawText()
      for reg in self.female_options.all_res:
        female_counts[reg] += len(self.female_options.all_res[reg].findall(story_text))
      for reg in self.male_options.all_res:
        male_counts[reg] += len(self.male_options.all_res[reg].findall(story_text))

    print 'Female counts:'
    print '=============='
    total = 0
    for reg in female_counts:
      print '%s: %d' % (reg, female_counts[reg])
      total += female_counts[reg]
    print '--------------'
    print 'Total:', total

    print
    print 'Male counts:'
    print '============'
    total = 0
    for reg in male_counts:
      print '%s: %d' % (reg, male_counts[reg])
      total += male_counts[reg]
    print '--------------'
    print 'Total:', total

if __name__ == '__main__':
  try:
    api_key = open('key.txt').read().strip()
    npr = NPR(api_key)

    # Expensive and *slow* (~5 min.)
    #npr.extractMatchingStories()

    npr.analyzeMatchingStories()
  except:
    keep_running = False
    raise
