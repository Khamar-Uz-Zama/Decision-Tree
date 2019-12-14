# Decision-Tree
Implementation of ID3 algorithm.

![Screenshot](screenshot.png)


Input parameters:
  --data: path to data
  --output: name of the output file

python3 decisiontree.py --data data.csv --output MyOutput.xml

The last column of the dataset must represent the class of the instance. 
The features of the dataset are unnamed but the output names them as 'attr' + column number

Packages like numpy/pandas are used for basic calculations, and ElementTree is used for creating the output xml file.
