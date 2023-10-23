# 1 Download the models.
```sh
python3 download_models.py
```

# 2.1 Run the demo on CPU
```sh
cd object_detection
# Take sample piture as input.
python3 main.py -i cars0.bmp
```

# 2.2 Run the demo on CPU
```sh
cd object_detection
# Take video stream from camera as input.
python3 main.py -i /dev/video0 -d /usr/lib/libethosu_delegate.so
```
