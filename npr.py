#!/usr/bin/env python

import urllib
import json

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

class Npr(object):
  baseUrl = 'http://api.npr.org/query?'

  def __init__(self, key):
    self.key_ = key

  def getUrl(self):
    getVars = {'apiKey': self.key_, 'numResults': 5, 'format': 'json'}
    return Npr.baseUrl + urllib.urlencode(getVars)

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
  npr = Npr(key)
  npr.getTopics()
