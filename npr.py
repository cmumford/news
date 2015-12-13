#!/usr/bin/env python

import copy
import csv
import datetime
import glob
import json
import re
import sys
import urllib
import xml.etree.ElementTree
import xml.etree.cElementTree as ET
import itertools
import dateutil.parser

# Uses the NPR API: http://api.npr.org/
# Query generator: http://www.npr.org/api/queryGenerator.php

# Last startNum retrieved was 154534

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
  def __init__(self, id):
    self.id_ = id
    self.title_ = ''
    self.date_ = None
    self.tags_ = []

class NPR(object):
  baseUrl = 'http://api.npr.org/query?'
  all_tags = []
  girl_tags = set()
  female_tags = set()
  boy_tags = set()
  male_tags = set()
  female_cancer_tags = set()
  male_cancer_tags = set()
  female_stories = set()
  male_stories = set()

  def __init__(self, api_key):
    self.api_key_ = api_key

  @staticmethod
  def loadTags():
    root = xml.etree.ElementTree.parse('tags.xml').getroot()
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
    ignore_ids = [126927651, 184560888]
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
    ignore_ids = [126927651, 184560888]
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
      149849695,
      149849693,
      135170830
    ]

  @staticmethod
  def findMenTags(all_tags):
    tags = set()
    ignore_ids = [126826632, 129251919, 152027155, 131877737]
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
    ignore_ids = [126826632, 129251919, 152027155, 131877737]
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
    return NPR.findMatchingTags(r'.*prostate cancer.*', all_tags)

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
      root = xml.etree.ElementTree.fromstring(xml_response)
      story_count = len(root.findall('list/story'))
      total_stories += story_count
      print 'there are', story_count, 'stories. So far:', total_stories
      params['startNum'] = params['startNum'] + story_count

  # Looks like searching by tags isn't an official API - I just guessed at it.
  # Apparently that query doesn't support startNum for pagination, so this
  # implementation uses dates to query the ranges.
  # Note: Note, looks like dates are also ignored - this not working either.
  def countTopics(self, tags):
    params = {'format':'json', 'fields': 'none'}
    if tags:
      params.update({'searchType':'tags', 'searchTerm':'|'.join(tags)})
    done = False
    story_count = 0

    start_date = datetime.datetime.strptime('2015-11-15', '%Y-%m-%d')
    end_date = datetime.datetime.now()
    while start_date < end_date:
      end = start_date + datetime.timedelta(days=6)
      p = copy.copy(params)
      p.update({'startDate':start_date.strftime('%Y-%m-%d'),
                'endDate':end.strftime('%Y-%m-%d')})
      url = self.getUrl(p)
      f = urllib.urlopen(url)
      json_obj = json.loads(f.read())
      story_list = json_obj['list']
      if 'story' in story_list:
        count = len(story_list['story'])
      else:
        count = 0
      print start_date, 'count:', count
      story_count += count
      start_date += datetime.timedelta(days=7)

    return story_count

  def loadStoriesFromFiles(self, all_tags, file_names):
    tags = {}
    for tag in all_tags:
      tags[tag.id_] = tag
    stories = []
    start = datetime.datetime.now()
    idx = 0
    for fname in file_names:
      idx += 1
      elapsed = datetime.datetime.now() - start
      files_per_sec = idx / elapsed.total_seconds()
      percent = idx * 100.0 / len(file_names)
      remaining_secs = (len(file_names) - idx) / files_per_sec
      print '%s: %.1f%%, fps:%.1f, remaining:%ds' % \
          (fname, percent, files_per_sec, remaining_secs)
      root = xml.etree.ElementTree.parse(fname).getroot()
      for xml_story in root.findall('list/story'):
        story = Story(int(xml_story.get('id')))
        story.title_ = xml_story.find('title').text
        story.date_ = dateutil.parser.parse(xml_story.find('storyDate').text)
        stories.append(story)
        for parent in xml_story.findall("parent[@type='tag']"):
          tag_id = int(parent.get('id'))
          tag = tags[tag_id]
          story.tags_.append(tag)
    return stories

  def loadStories(self, all_tags):
    file_names = glob.glob('stories/*.xml')
    return self.loadStoriesFromFiles(all_tags, file_names)

  def loadMatchingStories(self, tags, all_tags):
    stories = []
    for story in self.loadStories(all_tags):
      for tag in tags:
        if tag in story.tags_:
          stories.append(story)
          break
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

    tree = ET.ElementTree(root)
    tree.write("matching.xml")

  @staticmethod
  def loadTagsOfInterest():
    NPR.all_tags = NPR.loadTags()
    NPR.female_tags = NPR.findWomenTags(NPR.all_tags)
    NPR.female_cancer_tags = NPR.findWomenCancerTags(NPR.all_tags)
    NPR.male_tags = NPR.findMenTags(NPR.all_tags)
    NPR.male_cancer_tags = NPR.findMaleCancerTags(NPR.all_tags)
    NPR.boy_tags = NPR.findBoyTags(NPR.all_tags)
    NPR.girl_tags = NPR.findGirlTags(NPR.all_tags)

  # Extract a subset of the stories, and write them to a single file for
  # analysis.
  def extractMatchingStories(self):
    NPR.loadTagsOfInterest()

    combined_tags = [t for t in NPR.female_tags]
    combined_tags.extend([t for t in NPR.male_tags])

    matching_stories = npr.loadMatchingStories(combined_tags, NPR.all_tags)
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

  def analyzeMatchingStories(self):
    NPR.loadTagsOfInterest()
    stories = self.loadStoriesFromFiles(NPR.all_tags, ['matching.xml'])
    print 'There are', len(stories), 'matching'

    counts = NPR.calcTagCounts(stories, NPR.female_tags)
    NPR.printDictAsCSV(counts, 'analysis_female.xml')

    counts = NPR.calcTagCounts(stories, NPR.female_cancer_tags)
    NPR.printDictAsCSV(counts, 'analysis_female_cancer.xml')

    counts = NPR.calcTagCounts(stories, NPR.male_tags)
    NPR.printDictAsCSV(counts, 'analysis_male.xml')

    counts = NPR.calcTagCounts(stories, NPR.male_cancer_tags)
    NPR.printDictAsCSV(counts, 'analysis_male_cancer.xml')

    counts = NPR.calcTagCounts(stories, NPR.boy_tags)
    NPR.printDictAsCSV(counts, 'analysis_boys.xml')

    counts = NPR.calcTagCounts(stories, NPR.girl_tags)
    NPR.printDictAsCSV(counts, 'analysis_girls.xml')

if __name__ == '__main__':
  api_key = open('key.txt').read().strip()
  npr = NPR(api_key)

  #npr.extractMatchingStories()
  npr.analyzeMatchingStories()
