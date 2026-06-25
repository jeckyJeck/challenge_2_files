to use the data loader:

import the ImageClassificationDataModule class

dataclass = ImageClassificationDataModule(
        self,
        data_dir: str = DATA_DIR
        class_mapping_json: str = LABELS_JSON,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        val_split: float = 0.2,
        test_split: float = 0.0,        # if you want to use a test set, set this to a value between 0 and 1
        num_workers: int = 4,
        seed: int = 42
    )

now set the transforms you want to apply:

create list of transforms you want to apply, and pass it to the set_transforms() method of the dataclass.
here is a list from gemini to available transforms:
Geometric & Spatial Transforms
RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.33))

What it does: Crops a random area of the image and resizes it to size.

Parameters: size (target output size), scale (range of size of the origin size to be cropped), ratio (range of aspect ratio to be cropped).

RandomHorizontalFlip(p=0.5)

What it does: Horizontally flips the image randomly with a given probability.

Parameters: p (probability of the image being flipped).

RandomVerticalFlip(p=0.5)

What it does: Vertically flips the image randomly with a given probability.

Parameters: p (probability of the image being flipped).

RandomRotation(degrees, interpolation=InterpolationMode.NEAREST)

What it does: Rotates the image by a random angle.

Parameters: degrees (range of degrees to select from, e.g., (-15, 15)).

RandomAffine(degrees, translate=None, scale=None, shear=None)

What it does: Performs random affine transformations (rotation, translation, scale, and shear) keeping the center invariant.

Parameters: degrees (rotation range), translate (max absolute fraction for horizontal/vertical shifts), scale (scaling factor interval).

Color & Pixel-level Transforms
ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

What it does: Randomly changes the brightness, contrast, saturation, and hue of an image.

Parameters: Float values or tuples specifying the max jitter intensity for each factor.

RandomGrayscale(p=0.1)

What it does: Randomly converts the image to grayscale.

Parameters: p (probability that the image will be converted to grayscale).

GaussianBlur(kernel_size, sigma=(0.1, 2.0))

What it does: Blurs the image using a Gaussian filter.

Parameters: kernel_size (size of the blurring kernel), sigma (range of standard deviation for the Gaussian kernel).

RandomInvert(p=0.5)

What it does: Inverts the colors of the image randomly with a given probability.

Parameters: p (probability of the image being inverted).

RandomPosterize(bits, p=0.5)

What it does: Posterizes the image by reducing the number of bits for each color channel.

Parameters: bits (number of bits to keep for each channel, 0-8), p (probability).

Automated & Advanced Policies
RandAugment(num_ops=2, magnitude=9)

What it does: Automatically applies a random sequence of standard augmentations.

Parameters: num_ops (number of augmentation transformations to apply sequentially), magnitude (intensity magnitude for all transformations).

AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

What it does: Automatically applies augmentation policies learned on standard datasets.

Parameters: policy (the dataset policy to use: IMAGENET, CIFAR10, or SVHN).


and now you can get the train, validation and test dataloaders
with get_train_loader(), get_val_loader() and get_test_loader() methods.