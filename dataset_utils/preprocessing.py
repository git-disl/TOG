from PIL import Image
import numpy as np


def letterbox_image_padded(image, size=(416, 416)):
    """ Resize image with unchanged aspect ratio using padding """
    image_copy = image.copy()
    iw, ih = image_copy.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image_copy = image_copy.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (0, 0, 0))
    new_image.paste(image_copy, ((w - nw) // 2, (h - nh) // 2))
    new_image = np.asarray(new_image)[np.newaxis, :, :, :] / 255.
    meta = ((w - nw) // 2, (h - nh) // 2, nw + (w - nw) // 2, nh + (h - nh) // 2, scale)

    return new_image, meta
