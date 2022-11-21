# Color

## YUV
`YUV444`每一个Y对应一组UV分量。`YUV422`每两个Y共用一组UV分量。`YUV420`每四个Y共用一组UV分量。
```python
import cv2 as cv

cv.COLOR_YUV2BGR
cv.COLOR_YUV2BGR_I420
cv.COLOR_YUV2BGR_IYUV  => I420
cv.COLOR_YUV2BGR_NV12
cv.COLOR_YUV2BGR_NV21
cv.COLOR_YUV2BGR_UYNV
cv.COLOR_YUV2BGR_UYVY
cv.COLOR_YUV2BGR_Y422
cv.COLOR_YUV2BGR_YUNV
cv.COLOR_YUV2BGR_YUY2
cv.COLOR_YUV2BGR_YUYV
cv.COLOR_YUV2BGR_YV12
cv.COLOR_YUV2BGR_YVYU

cv.COLOR_BGR2YUV
cv.COLOR_BGR2YUV_I420
cv.COLOR_BGR2YUV_IYUV  => I420
cv.COLOR_BGR2YUV_YV12

cv.COLOR_YUV420P2BGR
cv.COLOR_YUV420SP2BGR
```

### YUV420
- YUV420P
  - U前V后: I420/YU12
  - V前U后: YV12
- YUV420SP
  - U前V后: NV12
  - V前U后: NV21

```text
      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y      Y Y Y Y Y Y
      U U U U U U      V V V V V V      U V U V U V      V U V U V U
      V V V V V V      U U U U U U      U V U V U V      V U V U V U
        - YU12 -         - YV12 -         - NV12 -         - NV21 -
```

先使用`ffmpeg`将指定的图片转为`YUV420P`格式
```sh
ffmpeg -i input.jpg -s 510x510 -pix_fmt yuv420p input.yuv
```
