# Testdata
This stores 3 evaluation files for automatic evaluation.
There is also manual testing, which will be executed on images in SampleData.
The testing will be run as 

```
python test.py --weights-file "outputs/x4/epoch50.pth" --image-file "testdata/SampleData/butterfly.jpg" --scale 2
```

To also save the image, run as:

```
python test.py --weights-file "outputs/x2/epoch50.pth" --image-file "testdata/SampleData/butterfly.jpg" --scale 2 --output-file testdata/SampleData/butterflysrx2.png
```
