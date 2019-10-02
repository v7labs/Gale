# fastSCSP
## A Fast Method for Inferring High-Quality Simply-Connected Superpixels
---------------------------------------------------------------------
This implementation is based on the algorithm from our paper, [\[Freifeld, Li and Fisher, ICIP '15\]](http://groups.csail.mit.edu/vision/sli/projects/fastSCSP/FreifeldLiFisher_ICIP15.pdf).
See also our [project page](http://groups.csail.mit.edu/vision/sli/projects.php?name=fastSCSP).

This software is released under the MIT License (included with the software).
Note, however, that if you use this code (and/or the results of running it) 
to support any form of publication (e.g.,a book, a journal paper, 
a conference paper, a patent application, etc.), then we ask you to cite
the following paper:

	@incollection{Freifeld:ICIP:2015,
	  title={A Fast Method for Inferring High-Quality Simply-Connected Superpixels},
	  author={Freifeld, Oren and Li, Yixin and Fisher III, John W},
	  booktitle={International Conference on Image Processing},
	  year={2015},
	}

Authors of this software: 
-------------------------

Yixin Li (email: liyixin@mit.edu)

Oren Freifeld  (email: freifeld@csail.mit.edu)

An early/partial version of this software, using python and CUDA, was written by Oren. It was then completed and improved by Yixin, who also wrote the Matlab and C++ wrappers. 

Versions
--------
07/27/2015, Version 1.03 -- Added a C++ wrapper (written by Yixin Li).

07/22/2015, Version 1.02 -- Added a Matlab wrapper (written by Yixin Li).

05/08/2015, Version 1.01 -- Minor bug fixes  (directory tree; relative paths; some windows-vs-linux issue) 

05/07/2015, Version 1.0  -- First release




Programming Languages
--------
Most of the computations are done in CUDA. 
We provide three (independent) wrappers:

1) Python;
2) Matlab;
3) C++.

The results/timings reported in our ICIP paper were obtained using the Python wrapper. 
The Matlab and C++ wrappers produce results that are very similar (but not 100% identical) to the results from the Python wrapper. While we have tested the Python wrapper extensively, we hardly tested the Matlab and C++ wrappers. However, they all wrap the same CUDA kernels.



While most of the work is done in CUDA, there are still some bookkeeping and minor computations that are done outside the CUDA kernels. Thus, as to be expected, the C++ wrapper is faster than the Python and Matlab versions. The python wrapper also sometimes halts a little the first time it is called (it seems to be less related to the generation of pyc files, and more related to argparse).

OS
--
All three wrappers were developed/tested on both Ubuntu 12.04 64-bit and Ubuntu 14.04 64-bit. 
The Python wrapper was also tested on Windows 7 Professional 64-bit. 
The Matlab and C++ wrapper should *probably* work on Windows.

Regarding Mac: we are fairly optimistic about it, but as we don't have access at the moment to a Mac with CUDA set up we couldn't test it. 

General Requirements 
--------------------

CUDA (version >= 5.5)
Additional Requirements for the C++ Wrapper
-----------------------------------------------
OpenCV

Additional Requirements for the Matlab Wrapper
-----------------------------------------------
Matlab's Parallel Computing Toolbox

Additional Requirements for the Python Wrapper
-----------------------------------------------
Numpy (version: developed/tested on 1.8. Some older versions should *probably* be fine too)

Scipy (version: developed/tested on 0.13.  Some older versions should *probably* be fine too)

matplotlib (version: developed/tested on 1.3.1.  Some older versions should *probably* be fine too)

pycuda (version: >= 2013.1.1)


Instructions for compiling the C++ wrapper
-----------------------------------------
The first thing you need to do is to specify the OpenCV_DIR in the CMakeLists.txt file.
On our machines this is "/usr/local/include/opencv"

Now run cmake (note the "."):

	 $ cmake .

Then run make:

	 $ make
This should generate two programs: Sp_demo and Sp_demo_for_direc 

Instructions for using the C++ wrapper
-----------------------------------------
Most of the instructions for the Python wrapper (see below) also apply to the C++ wrapper.
Alternatively, just take a look at the demo files in the cpp subdirectory.

Instructions for using the Matlab wrapper
-----------------------------------------
Most of the instructions for the Python wrapper (see below) also apply to the Matlab wrapper. 
Alternatively, just take a look at the demo files in the Matlab subdirectory.

Instructions for using the Python wrapper
-----------------------------------------
See demo.py for an example of running the algorithm on a single image.
See demo_for_direc.py for an example of running the algorithm on all files in a directory (this assumes that all files are in valid image format).

See the end of this README file for options the user can choose to speed up the initialization.





To run the algorithm on the default image (image/1.jpg) with default parameters:

	 python demo.py

To run on a user-specified image with default parameters:

	 python demo.py -i <img_filename>

For help:

	 python demo.py -h

To run on a user-specified image with user-specified parameters:

	 python demo.py -i <img_filename> -n <nPixels_on_side> --i_std <i_std>

In the initialization, the area of each superpixel is, more or less, nPixels_on_side^2. 
Let K denote the number of superpixels. High nPixels_on_side means small K and vice versa.

Note that the computation time *decreases* as the number of superpixels, K, increases.



The i_std controls the tradeoff between spatial and color features. A small i_std means a small standard deviation for the color features
(and thus making their effect more significant). In effect, small i_std = less regular boundaries.

To run the algorithm on all files in a user-specified directory:
Replace "demo.py" with "demo_for_direc.py" and replace "-i <img_filename>" with "-d <directory_name>"
The rest is the same as above.

Example 1: 
To run superpixel code on all images under default directory (./image):

	 python demo_for_direc.py
	 
Example 2: 

	 python demo_for_direc.py -d <directory_name>
  
Example 3: 

	 python demo_for_direc.py -d <directory_name> -n <nPixels_on_side> --i_std <i_std>
	 
Example 4: If all the images in the directory have the same size, you can save computing time by using

         python demo_for_direc.py -d <directory_name> --imgs_of_the_same_size 
         
(with or without the options for nPixels_on_side  and i_std mentioned above)

For help:

	 python demo_for_direc.py -h



The main functions in demo.py and demo_for_direc.py are:

1. Construct the SuperpixelsWrapper object (internally, this also initializes the segmentation according to the number of superpixels and image size):

	 sw = SuperpixelsWrapper(...)
	 
2. set the input image
  
	 sw.set_img(img)

3. compute the superpixels segmentation
  
	 sw.calc_seg(...)

4. copy parameters from gpu to cpu
  
	 sw.gpu2cpu()


Speeding up the initialization step
-----------------------------------

By default, we use an hexagonal honeycomb tiling, computed (on the GPU) using brute force. When K and/or the image is very large, this can be a bit slow.
Setting use_hex=False will use squares instead of hexagons. This will be faster, but less visually pleasing
(that said, in case you are obsessed with benchmarks, note we found that in comparison to hexagons, squares give slightly better results on benchmarks, although we didn't bother to include the square-based results in the paper). 
We thus suggest to stay with hexagons. To speed up the hexagonal initialization, 
we first try to load a precomputed initialization since it does not depend on the actual image. It depends only 
on the number of superpixels and the image size. If it doesn't exist, we compute and save it for a future use. Thus,
the next time the user calls the algorithm with the same image size and same K, it will be loaded instead of being computed from scratch.

As an aside remark, we actually played with several smarter options for computing the honeycomb tiling.
E.g., we used Warfield's K-DT algorithm (With K=1; his K does not have the same meaning as ours). 
However, while the complexity was fixed wrt (our) K, it was still slow for large images. Since loading precomputed results was much simpler and faster, we decided to go with that option.

Another step that takes some time but does not depend on the actual image is the construction of the SuperpixelWrapper object
(this will be faster once we move to a C++ wrapper). However, when running the algorithm on 
many images in the same directory where all images have the same size, this object needs to be constructed only once.  This is what the --imgs_of_the_same_size option mentioned above does, leading to some speedups.




