from __future__ import (absolute_import, division, print_function)
from PIL import Image
import six

import imagehash
def find_similar_images(userpaths, hashfunc = imagehash.average_hash):
    from PIL import Image
    import six
    import imagehash
    import tqdm
    import os
    
    def is_image(filename):
        f = filename.lower()
        return f.endswith(".png") or f.endswith(".jpg") or \
            f.endswith(".jpeg") or f.endswith(".bmp") or \
            f.endswith(".gif") or '.jpg' in f or  f.endswith(".svg")
    
    image_filenames = []
    for userpath in userpaths:
        image_filenames += [os.path.join(userpath, path) for path in os.listdir(userpath) if is_image(path)]
    images = {}
    for img in tqdm.tqdm(sorted(image_filenames)):
        try:
            hash = hashfunc(Image.open(img))
        except Exception as e:
            print('Problem:', e, 'with', img)
            continue
        if hash in images:
#             print(img, '  already exists as', ' '.join(images[hash]))
            if 'dupPictures' in img:
                print('rm -v', img)
        images[hash] = images.get(hash, []) + [img]
    return images