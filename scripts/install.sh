#! /bin/bash
cd libs/utils/libmesh
python setup.py install
cd ../MCAcc/cuda
python setup.py install