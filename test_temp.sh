xmake
xmake install
pip install ./python/
# python test/test_tensor.py

# python test/ops/argmax.py
# python test/ops/embedding.py
# python test/ops/linear.py
# python test/ops/rms_norm.py
python test/ops/rope.py 
python test/ops/self_attention.py
python test/ops/swiglu.py