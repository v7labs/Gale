#!/usr/bin/env python
"""
Created on Sat Sep  6 05:09:50 2014

Author: Oren Freifeld
Email: freifeld@csail.mit.edu
Generate the look up table for all 2^8 neighbor configurations,
which will be used in gpu/update_seg.cu
"""
import numpy as np


block0 = np.zeros((3,3),np.int)
block1 = np.zeros((3,3),np.int);block1[1,1]=1
nBits = 3
nhood=np.zeros((2**nBits,3))

def bin(num):   
    if not 0<=num<2**8:
        raise ValueError(num)    
    return  str(num) if num<=1 else bin(num>>1) + str(num&1)
    

def padbin(num):
    if not 0<=num<2**8:
        raise ValueError(num)
    tmp=bin(num)
    if len(tmp)==8:
        out=tmp
    else:
        out= '0'*(8-len(tmp))+tmp
    return out
    

def set_code_to_block(code,block):
    block.ravel()[:4]=map(int,code[:4])
    block.ravel()[5:]=map(int,code[4:])
    
#for i in range(256):   
#    code = padbin(i)
#    print '\n-------------'
#    print i,':',code
#    
#    set_code_to_block(code,block0)   
#    set_code_to_block(code,block1)   
#    
#    def find_ncc(block):
#        s=block.sum()
#        if s==0:
#            return 0
#        if s in (1,9):
#            return 1
#        elif s==2:
#            if  np.allclose(block0.sum(axis=1),[0,0,2]) and block0[:,1].any():
#                return 1
#            if  np.allclose(block0.sum(axis=1),[0,0,2]) and block0[:,1].any()==0:
#                return 2    
#        elif s==3:
#            if  np.allclose(block0.sum(axis=1),[0,0,3]):
#                return 1
#            
#        print block.astype(np.int)
#        raise NotImplementedError(i)
#    
#    print block0.astype(np.int)
#    print
#    ncc=find_ncc(block0)
#    print 'ncc =',ncc
#    

def check_code(code):
    """
    Return true if code is a simple-point.
    8-connectivity is used for BG 
    while 4-connectivity is used for FG.

    """
    set_code_to_block(code,block0)   
    set_code_to_block(code,block1)  
    
    if block0.any()==0:
        return False
    N = block0[0,1]
    S = block0[2,1]
    E = block0[1,2]
    W = block0[1,0]    

    NE = block0[0,2]
    NW = block0[0,0]
    SE = block0[2,2]
    SW = block0[2,0]
    
    b0sum = block0.sum()
    if b0sum==1:
        if N or S or E or W:
            return True
        else:
            return False
        
    if b0sum==2:
        if ((NW and N) or (NE and N) or (SW and S) or (SE and S)):
            return True
        if ((NW and W) or (NE and E) or (SW and W) or (SE and E)):
            return True        
        else:
            return False
    if b0sum==3:
        if (NW and N and NE) or (SW and S and SE): # lines: N and S
            return True
        if (NW and W and SW) or (NE and E and SE): # lines: W and E
            return True            
        if (NE and E and N) or (SE and E and S):  # Corners: NE and SE
            return True
        if (NW and W and N) or (SW and W and S):  # Corners: NW and SW
            return True
        return False
    if b0sum==4:
        # N  line 
        if (NW and N and NE) and (E or W):
            return True
        # S line 
        if (SW and S and SE) and (E or W):
            return True 
        # E line                     
        if (NE and E and SE) and (N or S):
            return True
        # W line                     
        if (NW and W and SW) and (N or S):
            return True 
        return False
    if b0sum==5:
        # S line 
        if (SW and S and SE) and (E and W):
            return True  
        # N line 
        if (NW and N and NE) and (E and W):
            return True  
        # E line
        if (NE and E and SE and N and S):
            return True
        # W line
        if (NW and W and SW and N and S):
            return True        
        # SE corner
        if (SW and S and SE and E and NE):
            return True
        # SW corner
        if (SW and S and SE and W and NW):
            return True
        # NE corner
        if (NW and N and NE and E and SE):
            return True
        # NW corner
        if (NW and N and NE and W and SW):
            return True        
        return False
    if b0sum==6:
        if (not NW) and ((not N) or (not W)):
            return True
        if (not NE) and ((not N) or (not E)):
            return True
        if (not SW) and ((not S) or (not W)):
            return True
        if (not SE) and ((not S) or (not E)):
            return True
        
        if (not NW) and (N or W):
            return False
        if (not NE) and (N or E):
            return False
        if (not SW) and (S or W):
            return False
        if (not N) and (NE or NW):
            return False
        if (not W) and (NW or SW):
            return False 
        if (not E) and (NE or SE):
            return False 
    if b0sum in (7,8):
        return True

 
def check_num(i,verbose=False):
    code = padbin(i)
    res=check_code(code)
    if verbose:         
        print '\n-------------'
        print 'Simple:',int(res)
        print i,':',code
        print block0
    return res
    
# generate the simple-point checking string    
L = [i for i in range(256) if check_num(i) ]
cond = '('+'||  '.join([' \b(num == {0})'.format(x) for x in L])+')' 
print cond

vals = ['{0}'.format(int(check_num(i))) for i in range(256)]
init_c_array = 'bool lut[256] = {' + ','.join(vals) + '};'
print init_c_array