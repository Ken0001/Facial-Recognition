"""
    Usage model is trained by https://github.com/bubbliiiing/facenet-pytorch.git
"""
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from PIL import Image
from facenet_pytorch.facenet import Facenet as facenet

class Facenet(object):
    _defaults = {
        "model_path"    : "models/facenet_inception_resnetv1.pth",
        "input_shape"   : [160, 160, 3],
        "backbone"      : "inception_resnetv1",
        "letterbox_image"   : True,
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.generate()
        self.show_config(**self._defaults)
        
    def show_config(self, **kwargs):
        print('Configurations:')
        print('-' * 70)
        print('|%25s | %40s|' % ('keys', 'values'))
        print('-' * 70)
        for key, value in kwargs.items():
            print('|%25s | %40s|' % (str(key), str(value)))
        print('-' * 70)
        
    def generate(self):
        print('Loading weights into state dict...')
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net    = facenet(backbone=self.backbone, mode="predict").eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()
            
    def resize_image(self, image, size, letterbox_image):
        iw, ih  = image.size
        w, h    = size
        if letterbox_image:
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            image   = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image
            
    def preprocess_input(self, image):
        image /= 255.0
        return image
    
    def inference(self, face):
        if isinstance(face, np.ndarray):
            # cv2
            import cv2
            face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        
        image = self.resize_image(face, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
        photo = torch.from_numpy(np.expand_dims(np.transpose(self.preprocess_input(np.array(image, np.float32)), (2, 0, 1)), 0))

        if self.cuda:
            photo = photo.cuda()
            
        output = self.net(photo).cpu().detach().numpy()
        return output
    
    """
    def detect_image(self, image_1, image_2):
        with torch.no_grad():
            image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
            image_2 = resize_image(image_2, [self.input_shape[1], self.input_shape[0]], letterbox_image=self.letterbox_image)
            
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(preprocess_input(np.array(image_2, np.float32)), (2, 0, 1)), 0))
            
            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
                
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()
            
            l1 = np.linalg.norm(output1 - output2, axis=1)
        
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(image_1))

        plt.subplot(1, 2, 2)
        plt.imshow(np.array(image_2))
        plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va= 'bottom',fontsize=11)
        plt.show()
        return l1
    """
    
if __name__ == "__main__":
    import cv2
    
    image1 = cv2.imread("dataset/Ken/S__24674311.jpg")
    image2 = cv2.imread("dataset/Ken/20230428_183803_safe.jpg")
    
    image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    
    model = Facenet()
    
    output1 = model.inference(image1)
    output2 = model.inference(image2)
    
    print(output1.shape)
    print(output2.shape)
    
    print("---")
    print(np.linalg.norm(output1 - output2, axis=1))
    print(np.linalg.norm(output1 - output2))