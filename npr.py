#!/usr/bin/env python

import json
import re
import urllib
import sys
import xml.etree.ElementTree

# Uses the NPR API: http://api.npr.org/
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

  def __init__(self, key):
    self.key_ = key

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
             'grandma']
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
             'grandpa']
    for word in words:
      reg = re.compile(r'.*\b%s\b.*' % word, re.IGNORECASE)
      for tag in all_tags:
        if tag.id_ not in ignore_ids and reg.match(tag.title_):
          tags.add(tag)
    return tags

  def getUrl(self, params = {}):
    common_params = {'apiKey': self.key_, 'format': 'json'}
    params.update(common_params)
    return NPR.baseUrl + urllib.urlencode(params)

  def getTopics(self):
    url = self.getUrl()
    f = urllib.urlopen(url)
    json_obj = json.loads(f.read())
    print json_obj['list']['title']['$text']
    for story in json_obj['list']['story']:
      print story['title']['$text']
      print story['teaser']['$text']
      print story['storyDate']['$text']
      for paragraph in story['text']['paragraph']:
        if '$text' in paragraph:
          print paragraph['$text']
      print "-------------------------------"

if __name__ == '__main__':
  key = open('key.txt').read().strip()
  npr = NPR(key)
  tags = NPR.loadTags()
  npr.getTopics()
