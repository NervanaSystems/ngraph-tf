Create a python 3 virtual environment

pip install the provided tensorflow whl file. 
`pip install -U /nfs/site/home/sarkars/onnx/tensorflow-1.12.0rc0-cp35-cp35m-linux_x86_64.whl`

Note: Do not use the default TF (`pip install tensorflow`). That does not have the DO quant ops support yet


Then run:
`python tf_dump.py config.json`


Edit `config.json` to specify desired output nodes
