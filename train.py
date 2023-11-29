import csv
import os
import cv2
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch import nn, optim
from torchsummary import summary
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from LoadDataset import LoadDataset
from VIT_Modules import ViTPose, ClassicDecoder
from Loss import AdaptiveWingLoss, pose_pck_accuracy, _get_max_preds
from Utility import tensor_to_image
from visualization import transform_preds


def train_ViTPose(
    model,
    train_loader,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    max_iters: int = 500,
    log_period: int = 20,
    num_epochs = 80,
    device: str = "cuda",
    model_save_path: str  =  " ",
    use_checkpoint = False,
    val = False ):

    """
    Train ViTmodel We use adamW optimizer and step decay.
    """
    
    model.to(device=device)
    HeatmapLoss =  AdaptiveWingLoss(use_target_weight=False)
    

    #   Optimizer: use adamW
    #   Use SGD with momentum:
    optimizer = optim.AdamW(model.parameters(), learning_rate, betas=(0.9, 0.999))
    
    avg_acc = 0
   
    if val:
        num_epochs = 1
        trained_model = torch.load(os.path.join(model_save_path,"model_params1_skirt.pth"))
        model.load_state_dict(trained_model)
    
    if use_checkpoint:
        print("LOADING pretrained WEIGHTS")
        trained_model = torch.load(os.path.join(model_save_path,"model_params1.pth"))
        model.load_state_dict(trained_model)
        print("Pretrained weights matched")
    try:
        lossFigure = []
        for epoch in range(num_epochs):




            list_bloust = []
            nnnn=0
            iterator = iter(train_loader)
            # Keep track of training loss for plotting.
            loss_history = []

            # detector.train()
            
            total_loss=torch.tensor(0.0).to(device)
            total_acc = 0

            for _iter in range(max_iters):
                # Ignore first arg (image path) during training.
                
                if not val:
                    images, target_heatmap, t_h_weight,_,_,_  = next(iterator)
                                    
                    t_h_weight = rearrange(t_h_weight, "B C H W ->  B H C W")
                    
                    images = images.to(device)
                    target_heatmap = target_heatmap.to(device)
                    t_h_weight =  t_h_weight.to(device)
                    
                    model.train()
                    model.zero_grad()

                    generated_heatmaps = model(images)
                    

                    # Dictionary of loss scalars.
                    losses = HeatmapLoss(generated_heatmaps, target_heatmap, t_h_weight)
                    total_loss+=losses

                    losses.backward()
                    optimizer.step()
                    ##确认target是否是需要的
                    _, avg_acc, _ = pose_pck_accuracy(generated_heatmaps.detach().cpu().numpy(),
                                                            target_heatmap.detach().cpu().numpy(),
                                                            t_h_weight.detach().cpu().squeeze(-1).numpy() > 0)
                    total_acc += avg_acc

                    if(_iter%log_period == 0 and _iter != 0):
                        _, avg_acc, _ = pose_pck_accuracy(generated_heatmaps.detach().cpu().numpy(),
                                                            target_heatmap.detach().cpu().numpy(),
                                                            t_h_weight.detach().cpu().squeeze(-1).numpy() > 0)
                        
                        loss_str = f"[Epoch {epoch}][ITER: {_iter}][loss: {losses:.8f}][Accuracy: {total_acc/(_iter+1):.8f}]" 
                        print(loss_str)
                        lossFigure.append(loss_str)


                else:
                    # VALIDATION
                    with torch.no_grad():
                        ##修改validation 看看在哪里出点的坐标 然后需要恢复啊 这点他变换过的

                        images, target_heatmap, t_h_weight,center, scale,paths   = next(iterator)
                        
                        
                        t_h_weight = rearrange(t_h_weight, "B C H W ->  B H C W")
                    
                        images = images.to(device)
                        target_heatmap = target_heatmap.to(device)
                        t_h_weight =  t_h_weight.to(device)
                        
                        model.eval()
                        generated_heatmaps = model(images)
                        nnnn = getBlouseForOutput(center, generated_heatmaps, images, list_bloust, nnnn, paths, scale)

                        #

                        #构造出来提交了 不管了
                        #transformed_kps就是想要的 全都可见把
                        _, avg_acc, _ = pose_pck_accuracy(generated_heatmaps.detach().cpu().numpy(),
                                                            target_heatmap.detach().cpu().numpy(),
                                                            t_h_weight.detach().cpu().squeeze(-1).numpy() > 0)
                    
                        total_acc += avg_acc
                        
                        if(_iter%log_period == 0 and _iter != 0): 

                            loss_str = f"[Epoch {epoch}][ITER: {_iter}][Accuracy: {total_acc/(_iter+1):.8f}]" 
                            print(loss_str)
                            lossFigure.append(loss_str)
            if not val:    
                # Print losses periodically.
                loss_str = f"[Epoch {epoch}][loss: {total_loss*images.size(0)/(max_iters):.8f}]"
                # for key, value in losses.items():
                #     loss_str += f"[{key}: {value:.3f}]"           
                print(loss_str)
                loss_history.append(total_loss.item())
                
            else:
                #list_blouse直接完成
                print(len(list_bloust))
                save_to_csv(list_bloust,r'C:\Users\28144\Desktop\gitcode\ViTPose-Implementation-main\Output\skirt.csv')
                # save_to_csv(list_bloust,
                #             r'C:\Users\28144\Desktop\gitcode\ViTPose-Implementation-main\Output\outwear.csv')
                # save_to_csv(list_bloust,
                #             r'C:\Users\28144\Desktop\gitcode\ViTPose-Implementation-main\Output\dress.csv')
                # save_to_csv(list_bloust,
                #             r'C:\Users\28144\Desktop\gitcode\ViTPose-Implementation-main\Output\skirt.csv')
                # save_to_csv(list_bloust,
                #             r'C:\Users\28144\Desktop\gitcode\ViTPose-Implementation-main\Output\trousers.csv')
                print(nnnn)
                print("Total_accuracy = ",total_acc/max_iters)
            
            if not val and (total_acc/max_iters) > 0.8:
                break
        if not val:
            print("------------model TRAINNED SUCESSFULLY------------")
            #这里是模型结束
            torch.save(model.state_dict(),os.path.join(model_save_path,"model_params1_skirt.pth"))
            with open(os.path.join(model_save_path,"loss.txt"), "w") as output:
                output.write(str(lossFigure))

    except KeyboardInterrupt:

        print("------------TRAINING SUSPENDED------------")
        if not val:
            torch.save(model.state_dict(),os.path.join(model_save_path,"model_params_skirt.pth"))
            with open(os.path.join(model_save_path,"loss.txt"), "w") as output:
                output.write(str(loss_history))


def save_to_csv(data, file_path):
    """
    将二维数组保存为CSV文件

    参数:
    - data: 二维数组，要保存的数据
    - file_path: 要保存的CSV文件的路径
    """
    with open(file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # 逐行写入数据
        a = True
        for row in data:
            csvwriter.writerow(row)
            if a:
                print(len(row))
                a = False

    print(f'CSV文件已保存到：{file_path}')


def dressTrans(listOnePic):
    listFinal = []
    # blouseTrans(listOnePic)
    # 0 1 2 3 4 5 6 7 8 9 10 11 12 0 0 0 0 17 18
    for info in range(len(listOnePic)):
        if (info < 15):
            listFinal.append(listOnePic[info])
        elif info ==15:
            listFinal.extend(["-1_-1_-1","-1_-1_-1","-1_-1_-1","-1_-1_-1"])
            listFinal.append(listOnePic[info])
        else:
            listFinal.append(listOnePic[info])
    # 26 15 +5 + 1 +5
    listFinal.extend(
        ["-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1"])
    return listFinal


def trousersTrans(listOnePic):
    listFinal = []
    # blouseTrans(listOnePic)
    # 15 16 19 20 21 22 23   2 3 4 5 6 7 8
    for info in range(len(listOnePic)):
        if (info < 2):
            listFinal.append(listOnePic[info])
        elif info == 2:
            listFinal.extend(["-1_-1_-1", "-1_-1_-1", "-1_-1_-1",
                              "-1_-1_-1","-1_-1_-1", "-1_-1_-1", "-1_-1_-1",
                              "-1_-1_-1","-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1","-1_-1_-1", "-1_-1_-1", "-1_-1_-1"])
            listFinal.append(listOnePic[info])
        elif info == 4:
            listFinal.extend(["-1_-1_-1", "-1_-1_-1"])
            listFinal.append(listOnePic[info])
        else:
            listFinal.append(listOnePic[info])
    # 26 9 + 15 + 2

    return listFinal


def skirtTrans(listOnePic):
    listFinal = []
    # blouseTrans(listOnePic)
    #  ## 15 16 17 18
    for info in range(len(listOnePic)):
        if (info < 2):
            listFinal.append(listOnePic[info])
        elif info == 2:
            listFinal.extend(["-1_-1_-1", "-1_-1_-1", "-1_-1_-1",
                              "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1",
                              "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1",
                              "-1_-1_-1"])
            listFinal.append(listOnePic[info])
        else:
            listFinal.append(listOnePic[info])
    # 26 6 + 15 + 5
    listFinal.extend(
        ["-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1"])
    return listFinal


def getBlouseForOutput(center, generated_heatmaps, images, list_bloust, nnnn, paths, scale):
    if (images.size(0) < 25):
        for imageSingle in range(images.size(0)):
            name1 = os.path.basename(paths[imageSingle])
            catgories = os.path.basename(os.path.dirname(paths[imageSingle]))
            tempOne = []
            listOnePic = []
            listOnePic.append("Images/" + catgories + "/" + name1)
            listOnePic.append(catgories)
            nnnn = nnnn + 1
            for num in range(4):
                max_coords_2d = divmod(torch.argmax(
                    torch.nn.functional.softmax(generated_heatmaps[imageSingle][num],
                                                dim=0)).item(),
                                       torch.nn.functional.softmax(
                                           generated_heatmaps[imageSingle][num], dim=0).shape[1])
                tempOne.append([max_coords_2d[0], max_coords_2d[1]])
            c = center[imageSingle]
            s = scale[imageSingle]
            transform_one = transform_preds(np.array(tempOne), c.squeeze().detach().numpy(),
                                            s.squeeze().detach().numpy(), heatmap_size)
            for kpp in transform_one:
                # 假设 kpp[0] 和 kpp[1] 是包含浮点数的 NumPy 数组
                kpp[0] = np.round(kpp[0], 2)
                kpp[1] = np.round(kpp[1], 2)

                # 将浮点数转换为字符串并进行相加
                res = kpp[0].astype(str) + "_" + kpp[1].astype(str) + "_1"
                listOnePic.append(res)
            # 换成24个的格式直接省的最后加东西
           # listFinal = blouseTrans(listOnePic)
            ##outwear
            #listFinal = outwearTrans(listOnePic)
            # listFinal = dressTrans(listOnePic)
            # listFinal = trousersTrans(listOnePic)
            listFinal = skirtTrans(listOnePic)
            list_bloust.append(listFinal)
    else:
        for imageSingle in range(25):
            name1 = os.path.basename(paths[imageSingle])
            catgories = os.path.basename(os.path.dirname(paths[imageSingle]))
            tempOne = []
            nnnn = nnnn + 1
            listOnePic = []
            listOnePic.append("Images/" + catgories + "/" + name1)
            listOnePic.append(catgories)
            for num in range(4):
                max_coords_2d = divmod(
                    torch.argmax(torch.nn.functional.softmax(generated_heatmaps[imageSingle][num], dim=0)).item(),
                    torch.nn.functional.softmax(generated_heatmaps[imageSingle][num], dim=0).shape[1])
                tempOne.append([max_coords_2d[0], max_coords_2d[1]])
            c = center[imageSingle]
            s = scale[imageSingle]
            transform_one = transform_preds(np.array(tempOne), c.squeeze().detach().numpy(),
                                            s.squeeze().detach().numpy(), heatmap_size)
            for kpp in transform_one:
                # 假设 kpp[0] 和 kpp[1] 是包含浮点数的 NumPy 数组
                kpp[0] = np.round(kpp[0], 2)
                kpp[1] = np.round(kpp[1], 2)
                # 将浮点数转换为字符串并进行相加
                res = kpp[0].astype(str) + "_" + kpp[1].astype(str) + "_1"
                listOnePic.append(res)
            # 处理listonePic
            # listFinal = outwearTrans(listOnePic)
            # listFinal = dressTrans(listOnePic)
            # listFinal = trousersTrans(listOnePic)
            listFinal = skirtTrans(listOnePic)
            list_bloust.append(listFinal)
    return nnnn


def outwearTrans(listOnePic):
    listFinal = []
    # blouseTrans(listOnePic)
    # 0 1 0 3 4 5 6 7 8 9 10 11 12 13 14 0 0 0 0 0 0 0 0 0
    for info in range(len(listOnePic)):
        if (info < 4):
            listFinal.append(listOnePic[info])
        elif info == 4:
            listFinal.extend(["-1_-1_-1"])
            listFinal.append(listOnePic[info])
        else:
            listFinal.append(listOnePic[info])
    # 26 4 + 2 + 11 +9
    listFinal.extend(
        ["-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1",
         "-1_-1_-1"])
    return listFinal


def blouseTrans(listOnePic):
    listFinal = []
    # blouse
    for info in range(len(listOnePic)):
        # listBlouse = [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]
        if info < 9:
            listFinal.append(listOnePic[info])
        elif info == 9:
            listFinal.extend(["-1_-1_-1", "-1_-1_-1"])
            listFinal.append(listOnePic[info])
        else:
            listFinal.append(listOnePic[info])
    # 26 9 +2 + 6 +9
    listFinal.extend(
        ["-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1", "-1_-1_-1",
         "-1_-1_-1"])
    return listFinal


if __name__ == "__main__":
    in_channels = 3
    patch_size = 16
    emb_size = 384#768
    img_size = (128,128)
    #img_size = (224, 192)
    heatmap_size = ((img_size[0]//patch_size)*4, (img_size[1]//patch_size)*4)
    depth = 6 #12                    #Depth of transformer layer
    kernel_size = (4,4)
    deconv_filter = 256
    #out_channels = 17

    train_dataset_size = 10000
    val_dataset_size = 6000
    batch_size = 25

    learning_rate = 2e-3
    weight_decay = 1e-4
    train_max_iters = train_dataset_size//batch_size
    val_max_iters = val_dataset_size//batch_size

    log_period = 20
    num_epochs = 80
    device = "cuda"
##需要修改
    out_channels = 4  ##
#soruce
    # img_directory = "D://coco//coco-2017-dataset//coco2017//images//train2017//"
    # annotation_path = "D://coco//coco-2017-dataset//coco2017//annotations//person_keypoints_train2017.json"
#blouse路径 训练
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\Images\blouse'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\annotations\blouse.json'
#blouse测试路径
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\Images\blouse'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\annotations\blouse.json'

#outwear
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\Images\outwear'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\annotations\outwear.json'
    # 测试路径
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\Images\outwear'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\annotations\outwear.json'

    # dress
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\Images\dress'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\annotations\dress.json'
    # 测试路径
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\Images\dress'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\annotations\dress.json'
    # trousers
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\Images\trousers'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\annotations\trousers.json'
    # 测试路径
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\Images\trousers'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\annotations\trousers.json'

    # skirt
    img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\Images\skirt'
    annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\train\annotations\skirt.json'
    # 测试路径
    # img_directory = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\Images\skirt'
    # annotation_path = r'C:\Users\28144\Desktop\TianChi-FashionAi-keypoints\fashionAi_Data\test\annotations\skirt.json'
    model_save_path = "C://Users//28144//Desktop//gitcode//ViTPose-Implementation-main//model"
    print(torch.__version__)
    print(torch.cuda.is_available())
    
    train = LoadDataset(img_directory, annotation_path,img_size, heatmap_size, test_mode = False)
    # indices = torch.arange(train_dataset_size)
    # train_30k= data_utils.Subset(train, indices)

    # print("TAKING SUBSET OF THE CURRENT DATASET")
    # print("New Dataset Size: ",train_30k.__len__())
    # train_loader = torch.utils.data.DataLoader(dataset=train_30k, 
    #                                             batch_size=batch_size,
    #                                             shuffle=True,
    #                                             num_workers=4)
        #img_size 不对 128*128 192 *225
    model = ViTPose(in_channels,patch_size,emb_size,img_size,depth,kernel_size,deconv_filter,out_channels)

    # train_ViTPose( model, train_loader, learning_rate, weight_decay, train_max_iters, log_period, num_epochs, device, model_save_path,use_checkpoint = True, val = False)
    
    #这个代码不好 用比例才能一直用
    #val_indices = torch.arange(9000, 10000)
    #val_10k= data_utils.Subset(train, val_indices)
    #train_dataset,val_dataset =

    val_loader = torch.utils.data.DataLoader(dataset=train,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=0,drop_last=False)
    print("Val Dataset Size: ",train.__len__())

    train_ViTPose( model, val_loader, learning_rate, weight_decay, -(-train.__len__()//batch_size), log_period, num_epochs, device, model_save_path,use_checkpoint = False,val = False)#true
    #print(summary(model,input_size=(in_channels, img_size[0], img_size[1])))
    # tensor, target, weight =  next(iter(val_loader))
    # # # weight = rearrange(weight, "B C H W ->  B H C W")
    # # print("Image size: ", tensor[0].shape)
    # # # print("heatmap size: ", target.shape)
    # # # print("Target weights: ",weight.shape)
    # image = tensor_to_image(tensor[5])
    # data = Image.fromarray(image)
    # data.show()
    # print(weight[5])
    
    
     

    
    
    
    





