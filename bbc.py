#!/usr/bin/env python3

import codecs
import glob
import os
import signal
import string
import sys
import npr

keep_running = True

def handler(signum, frame):
  print('Signal handler called with signal', signum)
  global keep_running
  if signum == signal.SIGINT:
    keep_running = False

class Text(object):
  exclude = set(string.punctuation)
  exclude.add(u'\u2013') # endash
  exclude.add(u'\u2014') # emdash
  excludeStr = ''.join(exclude)

  @staticmethod
  def stripPunctuation(text):
    return text.strip(Text.excludeStr)

  @staticmethod
  def extractWords(text):
    return [Text.stripPunctuation(w.strip()) for w in text.split()]

  @staticmethod
  def rawText(paragraphs):
    words = []
    for paragraph in paragraphs:
      newSentence = True
      for word in paragraph.split():
        word.strip()
        if newSentence:
          if len(word) > 1:
            word = word.lower()
          newSentence = False
        newSentence = word.endswith('.')
        words.append(Text.stripPunctuation(word))
    return ' '.join(words)

class Story(object):
  def __init__(self, category):
    self.category = category
    self.paragraphs = []
    self.title = ''

  def hasText(self):
    return len(self.paragraphs) > 0

  def rawText(self):
    return Text.rawText(self.paragraphs)

class BBC(object):
  def __init__(self):
    pass

  @staticmethod
  def readStory(file_name):
    category = os.path.basename(os.path.dirname(file_name))
    story = Story(category)
    sentence_num = 0
    with codecs.open(file_name, 'r', 'iso-8859-1') as f:
      for line in f.readlines():
        line = line.strip()
        if not line:
          continue
        sentence_num += 1
        if sentence_num == 1:
          story.title = line
        else:
          story.paragraphs.append(line)
      return story

  @staticmethod
  def readStories(dir):
    # Read the BBC story corpus downloaded
    # from http://mlg.ucd.ie/datasets/bbc.html
    stories = []
    file_names = glob.glob('%s/*/*.txt' % dir)
    for file_name in file_names:
      stories.append(BBC.readStory(file_name))
    return stories

  @staticmethod
  def classifyStories(stories):
    print("Predicting stories...")
    story_texts = []
    stories_being_classified = []
    for story in stories:
      if story.category == 'sport':
        continue
      if story.hasText():
        stories_being_classified.append(story)
        story_texts.append(story.rawText())
    if not story_texts:
      print('No story text to classify')
      return
    print('Predicting for', len(story_texts), 'non-sport stories')
    (clp, lb) = npr.NPR.createClassifierPipeline(18, apply_auto_tags=True)
    predicted = clp.predict(story_texts)

    new_labels = {}
    all_labels = lb.inverse_transform(predicted)
    num_labeled_stories = 0
    for story, categories in zip(stories_being_classified, all_labels):
      if len(categories):
        num_labeled_stories += 1
        for category in categories:
          if category in new_labels:
            new_labels[category] += 1
          else:
            new_labels[category] = 1
      print('%s => %s' % (story.title, ', '.join(categories) ))
    print()
    print('Labeled', num_labeled_stories, 'stories')
    print('Newly added tags: %d' % len(new_labels))
    print('====================')
    for category in sorted(new_labels.keys()):
      print('%s: %d' % (category, new_labels[category]))

if __name__ == '__main__':
  stories = BBC.readStories('bbc')
  print('Loaded', len(stories), 'stories')
  BBC.classifyStories(stories)
