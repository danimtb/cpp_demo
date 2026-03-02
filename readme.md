Build and run the demo:

```
cd cpp_demo
conan install --build missing -of build [-c tools.files.download:verify=False]
cmake --preset conan-release
cd build
cmake --build .
./depth_demo --video=../assets/video2.mp4
```

Download the model:

```
pip install -r requirements.txt
python download_model.py
```
