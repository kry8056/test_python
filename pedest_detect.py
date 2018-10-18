#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 14:07:14 2018

@author: ykarpitski
"""

datadir = "data"
dataset = "pedestrians128x64"
datafile = "%s/%s.tar.gz" % (datadir, dataset)

extractdir = "%s/%s" % (datadir, dataset)

def extract_tar(filename):
    try:
        import tarfile
    except ImportError:
        raise ImportError("You do not have tarfile installed. "
                      "Try unzipping the file outside of "
                      "Python.")
    tar = tarfile.open(datafile)
    tar.extractall(path=extractdir)
    tar.close()
    print("%s sucessfully extracted to %s" % (datafile, extractdir))
#Then we can call the function like this:
extract_tar(datafile)
