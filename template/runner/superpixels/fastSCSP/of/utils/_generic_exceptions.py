"""
Created on Tue Feb 11 09:53:27 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
"""
import os

def raise_value_error(x= None):
    if x is None:
        raise ValueError
    else:
        raise ValueError(x)


class PyGenericError(Exception):
    pass

class ShapeError(PyGenericError):
    pass

class ObsoleteError(PyGenericError):
    pass


class RawDataError(PyGenericError):
    def __init__(self,fname):
        self.value = 'filename (or dirname)"{0}" may stand for raw data!!!'.format(fname)
    pass
    def __str__(self):
        return repr(self.value) 


class DoesNotExistError(PyGenericError):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)
    
class AlreadyExistsError(PyGenericError):
    def __init__(self, name):
        self.value = ('{0} already exists.'.format(name))
    def __str__(self):
        return repr(self.value)


class FileAlreadyExistsError(AlreadyExistsError): pass
class DirAlreadyExistsError(AlreadyExistsError):  pass

class FileDoesNotExistError(DoesNotExistError): pass
class DirDoesNotExistError(DoesNotExistError):  pass





class Usage(PyGenericError):
    def __init__(self, msg):
        raise NotImplementedError
        self.msg = msg

   

