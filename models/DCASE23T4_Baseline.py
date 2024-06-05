import torch
import torch.nn as nn
from torchlibrosa.augmentation import SpecAugmentation


class my_CRNN(nn.Module):
    def __init__(self, classes_num, cnn_filters, rnn_hid, _dropout_rate):
        super(my_CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm1 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv2 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm2 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.conv3 = nn.Conv2d(in_channels=cnn_filters, out_channels=cnn_filters, kernel_size=(3, 3), padding='same')
        self.batch_norm3 = nn.BatchNorm2d(num_features=cnn_filters)
        
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 5))
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.dropout = nn.Dropout(_dropout_rate)

        self.gru1 = nn.GRU(int(6*cnn_filters), rnn_hid, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(rnn_hid*2, rnn_hid)
        self.spec_augmenter = SpecAugmentation(time_drop_width=3, time_stripes_num=6,
                                               freq_drop_width=3, freq_stripes_num=6)

        self.linear2 = nn.Linear(rnn_hid, classes_num)
        self.activate = nn.Sigmoid()
        
    def forward(self, input, specaug=False):
        
        if specaug:
            x = self.spec_augmenter(input[:,None,:,:])
        else:
            x = input[:,None,:,:]
        x = self.conv1(x)

        x = self.batch_norm1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        
        # Bidirectional layer
        recurrent, _ = self.gru1(x)
        x = self.linear1(recurrent)
        x = self.linear2(x)
        x = self.activate(x)
        return x

if __name__ == '__main__':
    model = my_CRNN(classes_num=11, cnn_filters=128, rnn_hid=64, _dropout_rate=0.2)
    inputs = torch.rand((32,100,128))
    out = model(inputs, True)
    import matplotlib.pyplot as plt
    plt.figure(num=1)
    plt.imshow(inputs[0].detach().numpy())
