import torch.nn as nn
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import os
import torch
import torchvision
import numpy as np
from PIL import Image


class MyConvTranspose2d(nn.Module):
    """
    Since transpose conv in pytorch doesn't garentee to have an output that has the exact same shape as the input feed to conv,
    so we have to specify the output shape
    ref: https://discuss.pytorch.org/t/the-output-size-of-convtranspose2d-differs-from-the-expected-output-size/1876/13
    """
    def __init__(self, conv, output_size):
        super(MyConvTranspose2d, self).__init__()
        self.output_size = output_size
        self.conv = conv

    def forward(self, x):
        x = self.conv(x, output_size=self.output_size)
        return x


class ModelVis:
    def __init__(self, model):
        self.model = model
        model_weights = []
        conv_layers = []
        model_children = list(model.children())
        counter = 0
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    child = model_children[i][j]
                    if type(child) == nn.Conv2d:
                        counter += 1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
        self.conv_layers = conv_layers
        self.conv_count = counter

    def disp_layer_img(self, image):
        image = image.clone()
        outputs = []
        names = []
        processed = []
        for layer in self.conv_layers[0:]:
            image = layer(image)
            outputs.append(image)
            names.append(str(layer))

        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map, 0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())
        fig = plt.figure(figsize=(30, 50))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 4, i + 1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(f"{names[i].split('(')[0]}_{i}", fontsize=30)

    @staticmethod
    def plot_block_images(images, title, fig_size=10):
        w = int(np.ceil(np.sqrt(len(images))))

        fig, ax = plt.subplots(w, w, figsize=(fig_size, fig_size))
        plt.axis('off')
        plt.subplots_adjust(left=0,
                            bottom=0,
                            right=1,
                            top=1,
                            wspace=0,
                            hspace=0)
        for idx, _filter in enumerate(images):
            row_idx, col_idx = idx // w, idx % w
            ax[row_idx, col_idx].imshow(_filter, cmap='gray')

            fig.suptitle(title, y=1.1)
        for i in range(w):
            for j in range(w):
                ax[i, j].axis('off')
        plt.show()

    def disp_filter_img(self, image, index=3, fig_size=10):
        image = image.clone()
        for n_layers in range(index + 1):
            image = self.conv_layers[0:][n_layers](image)

        filters = image.squeeze(0).detach().numpy()
        ModelVis.plot_block_images(filters, title=str(self.conv_layers[index]) + f' Depth: {index}', fig_size=fig_size)

    def disp_deconv(self, image, index, fig_size=10, channel_wise = True):
        seq = list(self.model.children())[0]
        counter = 0
        cur = 0
        for layer in seq:
            cur += 1
            if type(layer) == nn.Conv2d:
                counter += 1
            if counter >= index:
                break
        print(cur)
        pooling_indices = {}
        conv_shape = {}
        pool_shape = {}
        img_after_seq = image.clone()
        for idx, layer in enumerate(seq[:cur]):
            if type(layer) == nn.MaxPool2d:
                pool_shape[idx] = img_after_seq.shape
                maxpool_return_idx = nn.MaxPool2d(kernel_size=layer.kernel_size,stride=layer.stride, padding=layer.padding,
                                                  dilation=layer.dilation, ceil_mode=layer.ceil_mode, return_indices=True)
                img_after_seq, indices = maxpool_return_idx(img_after_seq)
                pooling_indices[idx] = indices

            elif type(layer) == nn.Conv2d:
                conv_shape[idx] = img_after_seq.shape #(img_after_seq.shape[-2], img_after_seq.shape[-1])
                img_after_seq = layer(img_after_seq)

            else:
                img_after_seq = layer(img_after_seq)

        img_array = []
        for channel in range(img_after_seq.shape[1]):
            img_backward = img_after_seq.clone()
            img_backward[0][:channel] = torch.zeros((channel,img_backward.shape[2], img_backward.shape[3]))
            img_backward[0][channel+1:] = torch.zeros((img_backward.shape[1]-channel-1, img_backward.shape[2], img_backward.shape[3]))

            for i in range(cur-1, -1, -1):

                layer = seq[i]
                if type(layer) == nn.Conv2d:

                    deconv = nn.ConvTranspose2d(in_channels=layer.out_channels, out_channels=layer.in_channels,
                                                kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                    my_deconv = MyConvTranspose2d(deconv, output_size=conv_shape[i])
                    img_backward = my_deconv(img_backward)

                elif type(layer) == nn.ReLU:
                    relu = nn.ReLU(inplace=False)
                    img_backward = relu(img_backward)

                elif type(layer) == nn.MaxPool2d:

                    maxunpool = nn.MaxUnpool2d(kernel_size=layer.kernel_size, stride=layer.stride,
                                               padding=layer.padding)

                    img_backward = maxunpool(img_backward, pooling_indices[i], output_size=pool_shape[i])
            img_backward = torchvision.transforms.Grayscale()(img_backward)
            img_backward = torchvision.transforms.ToPILImage()(img_backward.squeeze(0))
            img_array.append(img_backward)

        ModelVis.plot_block_images(img_array, str(self.conv_layers[index-1]) + f' Depth: {index-1}', fig_size=fig_size)

        # plt.title(str(self.conv_layers[index]) + f' Depth: {index}')
        #
        # if channel_wise:
        #     for channel in img_after_seq.detach().numpy().squeeze(0):
        #         plt.axis('off')
        #         plt.figure(figsize=(fig_size, fig_size))
        #         plt.imshow(channel, cmap='gray')
        #         plt.show()
        # else:
        #     img_after_seq_gray = torchvision.transforms.Grayscale()(img_after_seq)
        #     img_after_seq = torchvision.transforms.ToPILImage()(img_after_seq_gray.squeeze(0))
        #     # return img_after_seq
        #     plt.imshow(np.asarray(img_after_seq), cmap='gray')
        #     plt.show()




if __name__ == '__main__':
    pass
    # from data_utils import GenerateBackground, ResizeImageLoader
    # from models.models import alexnet
    # alex = alexnet(8, False)
    # alex.load_state_dict(torch.load('/Users/xuanmingcui/Downloads/fft_cur_fold_1/1-0.8.pt'),map_location=torch.device('cpu'))
    # transform = torchvision.transforms.Compose([
    #     torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406],
    #         std=[0.229, 0.224, 0.225]
    #     )
    # ])
    # img1 = Image.open(str(
    #     '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTrainingCNN/datasets/VOC2012_filtered/test/root/car/2008_000105.jpg'))
    # img1_T = transform(img1).unsqueeze(0)
    # anno_root = '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTrainingCNN/datasets/VOC2012_filtered/test/annotations'
    # background_generator = GenerateBackground(bg_type='color')
    # imgloader = ResizeImageLoader((150, 150), 0.8, anno_root, background_generator=background_generator)
    # img_loaded = imgloader(
    #     "/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTrainingCNN/datasets/VOC2012_filtered/test/root/car/2008_000105.jpg")
    # img_loaded_t = transform(img_loaded)
    # image = img_loaded_t.unsqueeze(0)
    #
    # alex_vis_1_8 = ModelVis(alex)
    # alex_vis_1_8.disp_deconv(1, 2)