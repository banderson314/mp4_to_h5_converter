# MP4 → HDF5 Converter for Ilastik Animal Tracking

A GUI tool for converting MP4 videos to HDF5 (`.h5`) format for use with
[Ilastik's Animal Tracking workflow](https://www.ilastik.org/documentation/tracking/tracking).

Ilastik does not natively support MP4 files. In addition to handling the
format conversion, this tool writes the vigra-style `axistags` attribute
that Ilastik requires to correctly identify the time axis in a dataset.

---

## Requirements

- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/)
- [h5py](https://pypi.org/project/h5py/)
- [NumPy](https://pypi.org/project/numpy/)
- tkinter (included with most Python installations)

```bash
pip install opencv-python h5py numpy