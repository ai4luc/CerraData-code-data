import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import VisionDataset

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#i'll try to use this if we have time. 
#that's not fully implemented now in this code so that's why i set dali_is_enabled False im both cases
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
    dali_is_enabled = False
except ImportError:
    dali_is_enabled = False

class CerradoDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super(CerradoDataset, self).__init__(root, transform=transform, target_transform=target_transform)

        filename = os.path.basename(root).lower() + '.pt'
        filepath = os.path.join(os.path.dirname(root), filename)

        if os.path.exists(filepath):
            dataset = torch.load(filepath)
            self.data = dataset['data']
            self.targets = dataset['targets']

        else:
            dataset = torchvision.datasets.ImageFolder(root)

            data = []
            targets = []
            with tqdm(total=len(dataset), ascii=True, desc=filename) as pbar:
                for i, (x, y) in enumerate(dataset):

                    if transform is not None:
                        x = transform(x)

                    if target_transform is not None:
                        y = target_transform(y)

                    x = self.pil_to_tensor(x)

                    data.append(x)
                    targets.append(y)

                    pbar.update(1)

            self.data = torch.stack(data, dim=0)
            self.targets = torch.tensor(targets)

            dataset = {'data': self.data, 'targets': self.targets}
            torch.save(dataset, filepath)

    def pil_to_tensor(self, pic):
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        return img

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class Subset(VisionDataset):
    def __init__(self, dataset, indices, transform=None, target_transform=None):
        super(Subset, self).__init__(root=dataset.root, transform=transform, target_transform=target_transform)

        if not isinstance(dataset, VisionDataset):
            RuntimeError("A VisionDataset must be passed.")

        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        img, target = self.dataset[self.indices[index]]

        if not self.transform is None:
            img = self.transform(img)

        if not self.target_transform is None:
            target =  self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.indices)

def get_loaders(run, args):
    dataset_path = os.path.join(args.root, 'cerradov3')
    dataset = CerradoDataset(dataset_path, transform=transforms.Resize((256, 256)))
   
    #create and save or load RANDOM splits
    if not os.path.exists(os.path.join(args.root, f'{run}_splits{args.ts}.npz')):
        print(f'Couldnt find any splits. create split for {run} run.')

        if args.ts == .8: # when len(train) != len(val)
            train_idx, testval_idx = train_test_split(np.arange(len(dataset)), test_size=.2, shuffle=True, stratify=dataset.targets)
            val_idx, test_idx = train_test_split(testval_idx, test_size=0.5, shuffle=True, stratify=[dataset.targets[x] for x in testval_idx])

        elif args.ts in [0.1, 0.05, 0.01, 0.002, 0.0015, 0.001, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]: # when len(train) == len(val)
            trainval_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=(1-2*args.ts), shuffle=True, stratify=dataset.targets)
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.5, shuffle=True, stratify=[dataset.targets[x] for x in trainval_idx])

        print('lenght of the partitions:', len(train_idx), len(val_idx), len(test_idx), sep=' ')
        np.savez(os.path.join(args.root, f'{run}_splits{args.ts}.npz'), train=train_idx, val=val_idx, test=test_idx)

    else:
        print(f'Found split for {run} run. Reading them ...')

        splits = np.load(os.path.join(args.root, f'{run}_splits{args.ts}.npz'))
        train_idx = splits['train']
        val_idx = splits['val']
        test_idx = splits['test']

        print('lenght of the partitions:', len(train_idx), len(val_idx), len(test_idx), sep=' ')
    
    if args.pretrained:
        dtype_change = transforms.ConvertImageDtype(torch.float)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    else:
        #for cerrado: 
        #X = np.transpose(image.img_to_array(img) / 255., (2, 0, 1))
        #torch.from_numpy(np.array([X]))
        dtype_change = transforms.ConvertImageDtype(torch.float)

        train_samples = dtype_change(dataset.data[torch.from_numpy(train_idx)])
        val_samples = dtype_change(dataset.data[torch.from_numpy(val_idx)])
        samples = torch.cat((train_samples, val_samples))

        normalize = transforms.Normalize(mean=torch.mean(samples, dim=[0,2,3]),
                                    std=torch.std(samples, dim=[0,2,3]))
        
    #transforms
    train_transform = [
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        dtype_change,
        normalize
    ]

    val_transform = [
        transforms.CenterCrop(224),
        dtype_change,
        normalize
    ]

    #train split
    train_dataset = Subset(dataset, train_idx, transform=transforms.Compose(train_transform))
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    #val split
    val_dataset = Subset(dataset, val_idx, transform=transforms.Compose(val_transform))
    val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size*8, shuffle=False, num_workers=args.workers, pin_memory=True)

    #test split
    test_dataset = Subset(dataset, test_idx, transform=transforms.Compose(val_transform))
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size*4, shuffle=False, num_workers=args.workers, pin_memory=True)

#    images = [train_dataset[0][0], val_dataset[0][0], test_dataset[0][0]]
#    for i, img in enumerate(images):
#        np_img = np.asarray(img)
#        plt.figure()
#        plt.imshow(np_img)
#        plt.savefig(f"sanity_check{run}{i}.png")
#        plt.close()

    return train_loader, val_loader, test_loader
