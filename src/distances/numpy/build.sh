python3 setup.py build_ext --inplace
mv distances.cpython* distances.so
python3 cdist.py
