
Build and run the demo:

```
cd cpp_demo
conan install --build missing -of build
cmake --preset conan-release
cmake --build build
cd build
./depth_demo
```


Download the model:

```
pip install -r requirements.txt
python download_model.py
```