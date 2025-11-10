import os
import numpy as np
import torch
import torch.utils.data as data
import argparse
from torchvision.transforms import ToTensor
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.autograd import Variable
from SEANet import sea_net
from PIL import Image

class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

class test_set(data.Dataset):
    def __init__(self, dataroot, finesize, result_path):
        super().__init__()
        self.root = dataroot
        self.fineSize = finesize
        self.result_path = result_path
        self.input_path = os.path.join(self.root, 'input')
        self.GT_path = os.path.join(self.root, 'GT')
        self.result_path = os.path.join(self.root, self.result_path)
        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)
        self.input_paths = sorted(make_dataset(self.input_path))
        self.GT_paths = sorted(make_dataset(self.GT_path))

        self.transform = ToTensor()

    def __getitem__(self, index):
        INPUT = Image.open(self.input_paths[index]).convert("RGB") 
        GT = Image.open(self.GT_paths[index]).convert("RGB") 
        resized_in = INPUT.resize((self.fineSize, self.fineSize), Image.Resampling.LANCZOS)
        resized_gt = GT.resize((self.fineSize, self.fineSize), Image.Resampling.LANCZOS)
        input = self.transform(resized_in)
        gt = self.transform(resized_gt)
        name = os.path.join(self.result_path, os.path.basename(self.input_paths[index]))

        return input, gt, name

    def __len__(self):
        return len(self.input_paths)

def compute_psnr_ssim(recoverd, clean):
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)
    recoverd = recoverd.transpose(0, 2, 3, 1)  
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0

    for i in range(recoverd.shape[0]):
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        ssim += structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis = -1)

    return psnr / recoverd.shape[0], ssim / recoverd.shape[0], recoverd.shape[0]

def test(args, model):
    
    model.eval()
    
    dataset = test_set(dataroot=args.input_root, finesize=args.size, result_path=args.result_path)
    data_load = data.DataLoader(dataset, batch_size=args.batchsize)

    test_psnr = AverageMeter()
    test_ssim = AverageMeter()
    
    for data_idx, data_ in enumerate(data_load):

        data_input, gt, img_name  = data_

        data_input = Variable(data_input).cuda()
        data_gt = Variable(gt).cuda()

        with torch.no_grad():
            result, _ = model(data_input)

            name = img_name[0]

            temp_res = np.transpose(result[0, :].cpu().detach().numpy(), (1, 2, 0))
            temp_res[temp_res > 1] = 1
            temp_res[temp_res < 0] = 0
            temp_res = (temp_res*255).astype(np.uint8)
            temp_res = Image.fromarray(temp_res)
            temp_res.save('%s' % (name))

            if args.test_score:
                temp_psnr, temp_ssim, N = compute_psnr_ssim(result, data_gt)  
                test_ssim.update(temp_ssim, N)
                test_psnr.update(temp_psnr, N)

                with open(args.score_file, 'a') as file:
                    file.write('{:<15}| PSNR: {:.4f}, SSIM: {:.4f}|\n'.format(os.path.basename(name), temp_psnr, temp_ssim))
    
    if args.test_score:
        with open(args.score_file, 'a') as file:
            file.write('total | PSNR: {:.4f}, SSIM: {:.4f}|\n'.format(test_psnr.avg, test_ssim.avg))

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Test')

    parser.add_argument('--batchsize', default=1, type=int, help='test batchsize')
    parser.add_argument('--size', default=256, type=int, help='crop size')
    parser.add_argument('--input_root', default='./test_img', type=str, help='data root path')
    parser.add_argument('--result_path',default='sea_pace_result', type=str)
    parser.add_argument('--score_file', default='sea_pace_score.txt',type=str)
    parser.add_argument('--model_root', default= './check_points/student.pth', type=str)
    parser.add_argument('--test_score', action= 'store_true')
    args = parser.parse_args()

    checkpoint = torch.load(args.model_root)

    model = sea_net().cuda()
    model.load_state_dict(checkpoint['state_dict'])

    test(args, model)

