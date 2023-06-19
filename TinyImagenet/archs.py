import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.model_zoo as model_zoo
import math
#from utee import misc
from collections import OrderedDict

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101','resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    # "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        m  = OrderedDict()
        m['conv1'] = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm2d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm2d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm2d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):#200):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=2, padding=3) 

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x) 
        x = self.maxpool(self.relu1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNetShared(ResNet):
    def forward(self, x):
        l = [] 
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        #print(x.shape)
        l.append(x)

        x = self.layer1(x)
        l.append(x)
        x = self.layer2(x)
        l.append(x)
        x = self.layer3(x)
        l.append(x)
        x = self.layer4(x)
        l.append(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x, l

def resnet18(pretrained=False, model_root=None, **kwargs):
    model = ResNetShared(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model.fc = nn.Linear(model.fc.in_features, 200)
    return model


def resnet34(pretrained=False, model_root=None, **kwargs):
    model = ResNetShared(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet34'], model_root)
    return model


def resnet50(pretrained=False, model_root=None, **kwargs):
    model = ResNetShared(Bottleneck, [3, 4, 6, 3], **kwargs)
    #model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet50'], model_root)
    return model


def resnet101(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet101'], model_root)
    return model


def resnet152(pretrained=False, model_root=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        misc.load_state_dict(model, model_urls['resnet152'], model_root)
    return model



class Decoder(nn.Module):

    def __init__(self, in_planes, final_upsample_mode = 'nearest'):
        super(Decoder, self).__init__()

        self.conv1x1_1 = self._make_conv1x1_upsampled(in_planes[0], 64)
        self.conv1x1_2 = self._make_conv1x1_upsampled(in_planes[1], 64, 2)
        self.conv1x1_3 = self._make_conv1x1_upsampled(in_planes[2], 64, 4)
        self.conv1x1_4 = self._make_conv1x1_upsampled(in_planes[3], 64, 8)
        self.cv = nn.Conv2d(64 + 4*64, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv1x1_upsampled(self, inplanes, outplanes, scale_factor=None):
        if scale_factor:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, l):
        k = []
        k.append(l[0])
        x = self.conv1x1_1(l[1])
        k.append(self.conv1x1_1(l[1]))
        x = self.conv1x1_2(l[2])
        k.append(self.conv1x1_2(l[2]))
        x = self.conv1x1_3(l[3])
        k.append(self.conv1x1_3(l[3]))
        x = self.conv1x1_4(l[4])
        k.append(self.conv1x1_4(l[4]))
        cated = self.cv(torch.cat(k, 1))
        cated = self.sigm(cated)
        return cated

def decoder18(**kwargs):
    return Decoder([64, 128, 256, 512], **kwargs)


class Decoder50(nn.Module):

    def __init__(self, in_planes, final_upsample_mode = 'nearest'):
        super(Decoder50, self).__init__()
        
        self.conv1x1_1 = self._make_conv1x1_upsampled(in_planes[0], 64)
        self.conv1x1_2 = self._make_conv1x1_upsampled(in_planes[1], 64, 2)
        self.conv1x1_3 = self._make_conv1x1_upsampled(in_planes[2], 64, 4)
        self.conv1x1_4 = self._make_conv1x1_upsampled(in_planes[3], 64, 8)
        self.final = nn.Sequential(
            nn.Conv2d(64 + 4*64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
            nn.Upsample(scale_factor=2, mode=final_upsample_mode)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_conv1x1_upsampled(self, inplanes, outplanes, scale_factor=None):
        if scale_factor:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, l):
        k = []
        k.append(l[0])
        k.append(self.conv1x1_1(l[1]))
        k.append(self.conv1x1_2(l[2]))
        k.append(self.conv1x1_3(l[3]))
        k.append(self.conv1x1_4(l[4]))
        return self.final(torch.cat(k, 1))

def decoder50(**kwargs):
    return Decoder50([64, 128, 256, 512], **kwargs)
    #return Decoder50([256, 512, 1024, 2048], **kwargs)


def create_models(args, device):
    print("=> creating models...")
    
    classifier = resnet18().to(device)
    #classifier = resnet50().to(device)
    decoder = decoder18(final_upsample_mode=args.upsample).to(device)
    optimizer = {}
    optimizer['classifier'] = torch.optim.SGD(classifier.parameters(), args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    optimizer['decoder'] = torch.optim.Adam(decoder.parameters(), args.lr_casme, weight_decay=args.weight_decay)
    
    return classifier, decoder, optimizer
