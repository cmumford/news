#!/usr/bin/env python

import copy
import datetime
import json
import re
import urllib
import sys
import xml.etree.ElementTree

# Uses the NPR API: http://api.npr.org/
# Query generator: http://www.npr.org/api/queryGenerator.php

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
  def findWomenTags(all_tags):
    tags = set()
    ignore_ids = []
    words = ['womens?', 'mothers?', 'girls?', 'daughters?', 'grandmothers?',
             'grandma', 'females?']
    for word in words:
      reg = re.compile(r'.*\b%s\b.*' % word, re.IGNORECASE)
      for tag in all_tags:
        if tag.id_ not in ignore_ids and reg.match(tag.title_):
          tags.add(tag)
    return tags

  @staticmethod
  def findMenTags(all_tags):
    tags = set()
    ignore_ids = [126826632, 129251919, 152027155]
    words = ['men', "men's", 'fathers?', 'boys?', 'sons?', 'grandfathers?',
             'grandpa', 'male?']
    for word in words:
      reg = re.compile(r'.*\b%s\b.*' % word, re.IGNORECASE)
      for tag in all_tags:
        if tag.id_ not in ignore_ids and reg.match(tag.title_):
          tags.add(tag)
    return tags

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

if __name__ == '__main__':
  api_key = open('key.txt').read().strip()
  npr = NPR(api_key)
  tags = NPR.loadTags()
  print 'There are', len(tags), 'total tags'
  if False:
    for tag in tags:
      print tag
  filtered = NPR.findWomenTags(tags)
  print 'There are', len(filtered), 'filtered tags'
  str_tags = []
  for tag in filtered:
    print tag
    str_tags.append(str(tag.id_))
  print 'There are', npr.countTopics(str_tags), 'stories'
