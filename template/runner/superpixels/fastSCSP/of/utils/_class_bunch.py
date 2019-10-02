#!/usr/bin/env python
"""
Created on Tue Sep 25 15:06:05 2012

Author: Oren Freifeld
Email: freifeld@dam.brown.edu
"""
 
class Bunch(object):
    """
    A thin wrapper to a dict.
    Goal: To enable easy access (e.g.; d.x instead of d['x'])
    Based on http://code.activestate.com/recipes/52308/
    but with some differences and additions. For example, 
    I would rather avoid inheriting from dict,
    since for convenient code-completion, I don't want
    to see all the attributes of dict when I press tab. I just
    want to see the keys.
    """
    def __init__(self, **kw):
        self.__dict__.update(kw) 
    def __len__(self):
        return len(self.__dict__)  
    def keys(self):
        return self.__dict__.keys()
    def __repr__(self):
        return repr(self.__dict__)                
    def __iter__(self):
        return iter(self.__dict__)
    def __getitem__(self,y):
        return self.__dict__.__getitem__(y)
    def __setitem__(self,i,y):
        return self.__dict__.__setitem__(i, y)
            
    def items(self):
        return self.__dict__.items()
    def iteritems(self):
        return self.__dict__.iteritems()        
    def itervalues(self):
        return self.__dict__.values()
    
if __name__ == '__main__':
#    b = Bunch(**{'k1':1,'k2':2})
    d = {'k1':1,'k2':2}
    b = Bunch(k1=2,k2=3)
    print(b)    
    
    for x in b:
        print(x)
    print 
    for k,v  in b.items():
        print('b.k = ',v)   
    print
    for k,v in b.iteritems():
        print('b.k = ',v)     
    print
    for x in b.itervalues():
        print(x)       
