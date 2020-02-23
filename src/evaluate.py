# -*- coding: utf-8 -*-

#from __future__ import print_function

from scipy import spatial
from PIL import Image
from statistics import mode
from sklearn.utils.extmath import weighted_mode
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class Evaluation(object):

  def make_samples(self):
    raise NotImplementedError("Needs to implemented this method")


def distance(v1, v2, d_type='d1'):
  assert v1.shape == v2.shape, "shape of two vectors need to be same!"

  if d_type == 'd1':
    return np.sum(np.absolute(v1 - v2))
  elif d_type == 'd2':
    return np.sum((v1 - v2) ** 2)
  elif d_type == 'd2-norm':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd3':
    pass
  elif d_type == 'd4':
    pass
  elif d_type == 'd5':
    pass
  elif d_type == 'd6':
    pass
  elif d_type == 'd7':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'd8':
    return 2 - 2 * np.dot(v1, v2)
  elif d_type == 'cosine':
    return spatial.distance.cosine(v1, v2)
  elif d_type == 'square':
    return np.sum((v1 - v2) ** 2)


def AP(label, results, sort=True):
  ''' infer a query, return it's ap

    arguments
      label  : query's class
      results: a dict with two keys, see the example below
               {
                 'dis': <distance between sample & query>,
                 'cls': <sample's class>
               }
      sort   : sort the results by distance
  '''
  if sort:
    results = sorted(results, key=lambda x: x['dis'])
  precision = []
  hit = 0
  for i, result in enumerate(results):
    if result['cls'] == label:
      hit = 1
      precision.append(hit)
 
  return (np.mean(precision)>0.5)


def infer(query, samples=None, db=None, sample_db_fn=None, depth=None, d_type='d1'):
  ''' infer a query, return it's ap

    arguments
      query       : a dict with three keys, see the template
                    {
                      'img': <path_to_img>,
                      'cls': <img class>,
                      'hist' <img histogram>
                    }
      samples     : a list of {
                                'img': <path_to_img>,
                                'cls': <img class>,
                                'hist' <img histogram>
                              }
      db          : an instance of class Database
      sample_db_fn: a function making samples, should be given if Database != None
      depth       : retrieved depth during inference, the default depth is equal to database size
      d_type      : distance type
  '''
  assert samples != None or (db != None and sample_db_fn != None), "need to give either samples or db plus sample_db_fn"
  if db:
    samples = sample_db_fn(db)

  q_img, q_cls, q_hist = query['img'], query['cls'], query['hist']
  results = []
  for idx, sample in enumerate(samples):
    s_img, s_cls, s_hist = sample['img'], sample['cls'], sample['hist']
    if q_img == s_img:
      continue
    results.append({
                    'img':s_img,
                    'dis': distance(q_hist, s_hist, d_type=d_type),
                    'cls': s_cls
                  })
  results = sorted(results, key=lambda x: x['dis'])
  if depth and depth <= len(results):
    results = results[:depth]
    print("image modèle: ",q_img)

    list_im = [ sub['img'] for sub in results]
    print(list_im)
    
    pred = [sub['cls'] for sub in results ]
    weight = [sub['dis'] for sub in results ]
    weight = np.reciprocal(weight)
    #pred = mode(pred)
    pred2 = weighted_mode(pred, weight)
    pred = np.array_str(pred2[0])[2:-2]
    
    list_im.insert(0, q_img)
    imgs = [Image.open(i) for i in list_im]
    
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgs))
    
    plt.imshow(imgs_comb/255.)
    plt.pause(2)
    plt.close()
    
   # ap n'est plus nécessaire quand toutes les images sont dans le meme dossier
  ap = AP(q_cls, results, sort=False)

  return ap, pred


def evaluate(db, db2, sample_db_fn, depth=None, d_type='d1'):
  ''' infer the whole database

    arguments
      db          : an instance of class Database
      sample_db_fn: a function making samples, should be given if Database != None
      depth       : retrieved depth during inference, the default depth is equal to database size
      d_type      : distance type
  '''
  
  samples = sample_db_fn(db)
  print(len(samples))
  
  samples2 = sample_db_fn(db2)
  print(len(samples2))
  
  classes = db.get_class()
  classes.add(samples2[0]['cls'])
  print(classes)
  
  ret = {c: [] for c in classes}
  
  predict = []
  
  i=0
  for query in samples2:
    i+=1
    print(i)
    ap, pred = infer(query, samples=samples, depth=depth, d_type=d_type)
    #ap, _ = infer(query, samples=samples, depth=depth, d_type=d_type)
    ret[query['cls']].append(ap)
    predict.append(pred)

  return ret, predict
  #return ret



