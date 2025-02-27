try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
    from nvidia.dali.pipeline import pipeline_def
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

import os
import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2

TRAIN_CROP = 224
VAL_CROP = 256



@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True, is_val=False):
    if is_val:
        images, labels = fn.readers.file(file_list=data_dir, file_root=os.path.join(data_dir, "images"),
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
    else:
        images, labels = fn.readers.file(file_root=data_dir,
                                        shard_id=shard_id,
                                        num_shards=num_shards,
                                        random_shuffle=is_training,
                                        pad_last_batch=True,
                                        name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    assert(dali_device != "cpu")
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               #random_aspect_ratio=[0.8, 1.25], # random_area=[0.08, 1.0]
                                               random_aspect_ratio = [0.75, 1.33],
                                               random_area=[0.08, 1.0], #random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        # in pytrochvision transforms.ColorJitter(brightness, contrast, saturation, hue) is equivalent to
        # brightness_factor = uniformly from [max(0, 1 - brightness), 1 + brightness]
        # contrast_factor   = uniformly from [max(0, 1 - contrast), 1 + contrast]
        # saturation_factor = uniformly from [max(0, 1 - saturation), 1 + saturation]
        # hue_factor        = uniformly from [-hue, hue]
        # we need an equivalent of transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        # twist = fn.color_twist(device="gpu")
        brightness = fn.random.uniform(range=[1-0.2, 1+0.2])
        saturation = fn.random.uniform(range=[1-0.2, 1+0.2])
        contrast = fn.random.uniform(range=[1-0.2, 1+0.2])
        hue = fn.random.uniform(range=[-0.1, 0.1])
        images = fn.color_twist(images, device=dali_device,
                                saturation=saturation, contrast=contrast,
                                brightness=brightness, hue=hue)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels

def get_val_transform():
    return v2.Compose([v2.Resize((VAL_CROP, VAL_CROP)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_general_transform(trans):
    return v2.Compose([v2.RandomResizedCrop((TRAIN_CROP, TRAIN_CROP)),
        v2.RandomHorizontalFlip(0.5),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.1),
        trans,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_augmix_transform():
    auto = v2.AugMix()
    return get_general_transform(auto)

def get_trivial_transform():
    trivial = transforms.TrivialAugmentWide()
    return get_general_transform(trivial)

def get_rand_transform():
    rand = v2.RandAugment()
    return get_general_transform(rand)

def get_erasing_transform():
    erasing = v2.RandomErasing()
    return get_general_transform(erasing)

def get_imagenet_aa_transform():
    aa = v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)
    return get_general_transform(aa)

def get_svhn_aa_transform():
    aa = v2.AutoAugment(v2.AutoAugmentPolicy.SVHN)
    return get_general_transform(aa)

class HFDataset(Dataset):
    def __init__(self, ds, transform):
        super().__init__()
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        img = self.ds[index]["image"]
        label = self.ds[index]["label"]
        img = self.transform(img)
        return img, label
    

def build_train_transform(name):
    if isinstance(name, str):
        if name == "auto":
            trans = v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)
        elif name == "trivial":
            trans = transforms.TrivialAugmentWide()
        elif name == "augmix":
            trans = v2.AugMix()
        elif name == "rand":
            trans = v2.RandAugment()
        elif name == "erasing":
            trans = v2.RandomErasing()
        elif name == "autoimg":
            trans = v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)
        elif name == "autosvhn":
            trans = v2.AutoAugment(v2.AutoAugmentPolicy.SVHN)
        elif name == "none":
            trans = v2.Identity()
        else:
            return NotImplementedError
    else:
        return NotImplementedError
    return get_general_transform(trans)


class HFDatasetMulti(Dataset):
    def __init__(self, ds, transform):
        super().__init__()
        self.ds = ds
        self.transform = transform
        self.basic_transform = build_train_transform("none")

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        img_raw = self.ds[index]["image"]
        label = self.ds[index]["label"]
        img = self.transform(img_raw)
        img2 = self.basic_transform(img_raw)
        return img, img2, label

def imagenet_data_loader(args, multi=False):
    train_ds = datasets.load_dataset("zh-plus/tiny-imagenet", split="train")
    val_ds = HFDataset(datasets.load_dataset("zh-plus/tiny-imagenet", split="valid"), get_val_transform())
    if multi:
        train_ds = HFDatasetMulti(train_ds, build_train_transform(args.transform))
    else:
        train_ds = HFDataset(train_ds, build_train_transform(args.transform))

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.workers)
    # train_loader =  
    # print(f"Data Folder: {args.data}")
    # if len(args.data) == 1:
    #     traindir = os.path.join(args.data[0], 'train')
    #     valdir = os.path.join(args.data[0], 'val')
    # else:
    #     traindir = args.data[0]
    #     valdir= args.data[1]


    # crop_size = 224
    # val_size = 256

    # pipe = create_dali_pipeline(batch_size=args.batch_size,
    #                             num_threads=args.workers,
    #                             device_id=args.local_rank,
    #                             seed=12 + args.local_rank,
    #                             data_dir=traindir,
    #                             crop=crop_size,
    #                             size=val_size,
    #                             dali_cpu=args.dali_cpu,
    #                             shard_id=args.local_rank,
    #                             num_shards=args.world_size,
    #                             is_training=True)
    # pipe.build()
    # train_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    # val_label = os.path.join(valdir, "val_annotations.txt")
    # pipe = create_dali_pipeline(batch_size=args.batch_size,
    #                             num_threads=args.workers,
    #                             device_id=args.local_rank,
    #                             seed=12 + args.local_rank,
    #                             data_dir=val_label,
    #                             crop=crop_size,
    #                             size=val_size,
    #                             dali_cpu=args.dali_cpu,
    #                             shard_id=args.local_rank,
    #                             num_shards=args.world_size,
    #                             is_training=False, is_val=True)
    # pipe.build()
    # val_loader = DALIClassificationIterator(pipe, reader_name="Reader", last_batch_policy=LastBatchPolicy.PARTIAL)
    return train_loader, val_loader