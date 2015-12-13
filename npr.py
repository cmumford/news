#!/usr/bin/env python

import copy
import datetime
import glob
import json
import re
import sys
import urllib
import xml.etree.ElementTree

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

class NPR(object):
  baseUrl = 'http://api.npr.org/query?'
  cancer = True

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
             'ovarian transplant']
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
  def femaleTags(all_tags):
    if NPR.cancer:
      return NPR.findWomenCancerTags(all_tags)
    else:
      return NPR.findWomenTags(all_tags)

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
  def findMaleCancerTags(all_tags):
    return NPR.findMatchingTags(r'.*prostate cancer.*', all_tags)

  @staticmethod
  def maleTags(all_tags):
    if NPR.cancer:
      return NPR.findMaleCancerTags(all_tags)
    else:
      return NPR.findMenTags(all_tags)

  def getUrl(self, params = {}):
    common_params = {'apiKey': self.api_key_}
    params.update(common_params)
    return NPR.baseUrl + urllib.urlencode(params)

  @staticmethod
  def getYMD(dt):
    return dt.strftime('%Y-%m-%d')

  def downloadData(self):
    params = {'startNum':30640, 'numResults':20}
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

  def countStories(self, tags):
    counts = {}
    ids_to_tag = {}
    for tag in tags:
      counts[tag] = 0
      ids_to_tag[tag.id_] = tag
    start = datetime.datetime.now()
    files = glob.glob('stories/*.xml')
    idx = 0
    for fname in files:
      idx += 1
      elapsed = datetime.datetime.now() - start
      files_per_sec = idx / elapsed.total_seconds()
      percent = idx * 100.0 / len(files)
      remaining_secs = (len(files) - idx) / files_per_sec
      print '%s: %.1f%%, fps:%.1f, remaining:%ds' % \
          (fname, percent, files_per_sec, remaining_secs)

      root = xml.etree.ElementTree.parse(fname).getroot()
      for story in root.findall('list/story'):
        for parent in story.findall("parent[@type='tag']"):
          tag_id = int(parent.get('id'))
          if tag_id in ids_to_tag:
            tag = ids_to_tag[tag_id]
            counts[tag] = counts[tag] + 1
    return counts

if __name__ == '__main__':
  api_key = open('key.txt').read().strip()
  npr = NPR(api_key)
  tags = NPR.loadTags()
  print 'There are', len(tags), 'total tags'
  female_tags = NPR.femaleTags(tags)
  print 'There are', len(female_tags), 'female tags'
  male_tags = NPR.maleTags(tags)
  print 'There are', len(male_tags), 'male tags'

  counts = npr.countStories(tags)

  female_count = 0
  male_count = 0
  for tag in counts:
    if tag in female_tags:
      female_count += counts[tag]
    if tag in male_tags:
      male_count += counts[tag]

  print 'There are', female_count, 'stories with female tags'
  print 'There are', male_count, 'stories with male tags'
