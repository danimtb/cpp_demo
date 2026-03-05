Build and run the demo:

```
cd cpp_demo
conan install --build missing [-c tools.files.download:verify=False]
cmake --preset conan-release
cmake --build build/Release
cd build/release
./depth_demo [--video=../../assets/video2.mp4] [--model=../../models/MiDaS_small.pt]
```

Download the model:

```
pip install -r requirements.txt
python download_model.py
```
