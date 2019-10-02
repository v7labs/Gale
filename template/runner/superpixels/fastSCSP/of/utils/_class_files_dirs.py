#!/usr/bin/env python
"""
Created on Thu Jun 28 11:05:25 2012

Author: Oren Freifeld
Email: freifeld@dam.brown.edu
"""
import inspect
from ._generic_exceptions import *
_isfile = os.path.isfile
_isdir = os.path.isdir


class FilesDirs:               
    @classmethod
    def raise_if_dir_does_not_exist(cls,dirname):
        _isdir(dirname) or cls._raise_dir_does_not_exist(dirname)
    @classmethod
    def _raise_file_does_not_exist(cls,filename):
        raise FileDoesNotExistError(filename)
    @classmethod
    def raise_if_file_does_not_exist(cls,filename):
        _isfile(filename) or cls._raise_file_does_not_exist(filename)
    @classmethod
    def _raise_file_already_exists(cls,filename):
        raise FileAlreadyExistsError(filename)
    @classmethod
    def _raise_dir_does_not_exist(cls,dirname):        
        raise DirDoesNotExistError(dirname)
    @classmethod
    def _raise_dir_already_exists(cls,dirname):
        raise DirAlreadyExistsError(dirname)
    @classmethod    
    def raise_if_dir_already_exists(cls,dirname):
        _isdir(dirname)==False or cls._raise_dir_already_exists(dirname) 
        _isfile(dirname)==False or cls._raise_file_already_exists(dirname)
           
    @classmethod 
    def raise_if_file_already_exists(cls,filename):
        _isfile(filename)==False or cls._raise_file_already_exists(filename) 
        _isdir(filename)==False or cls._raise_dir_already_exists(filename)

    @classmethod
    def verify_correct_file_ext(cls,filename,exts):
        ext = os.path.splitext(filename)[1]
         
        if not ext:        
            raise ValueError("""
            I cowardly refuse to guess the filetype for '{0}'. 
            Please provide file extension as in 
            <blahblah.ext> 
            or 
            <blahblah/.../blahblha/.ext>.""".format(filename))
        if isinstance(exts,str):
            exts = [exts]
        #Add periods if needed
        exts = [e if e.startswith('.') else '.'+e for e in exts]
        exts = [e.lower() for e in exts]
        del e                        
        ext = ext.lower()
        if not ext in exts:
            raise ValueError('{0} not in {1}'.format(ext,exts))


    @classmethod
    def mkdirs_if_needed(cls,dirname,verbose=False):
        try:
            cls.raise_if_dir_does_not_exist(dirname) 
        except DirDoesNotExistError:
            if verbose:
                print('mkdir ',dirname) 
            os.makedirs(dirname)
            

    @classmethod
    def filename_has_this_extension(cls,filename,ext):
        _ext = os.path.splitext(filename)[1]
        if ext.startswith('.'):
            return ext == _ext
        return ext == _ext[1:]    
        
        
    @classmethod
    def get_sorted_list_of_all_files_in_directory(cls,dirname):
        cls.raise_if_dir_does_not_exist(dirname)
        return sorted(os.listdir(dirname))


    @staticmethod
    def strip_filename_from_path_and_ext(filename):
        return os.path.splitext(os.path.split(filename)[-1])[0]
