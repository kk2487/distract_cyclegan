import os
import sys
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im
from data.base_dataset import get_transform 

import PIL
from PIL import Image, ImageOps
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import PyQt5.QtGui as QtGui
from PyQt5.QtCore import *
import cv2
import numpy as np
import time

import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import transforms
from FacePose_pytorch.dectect import AntiSpoofPredict
from FacePose_pytorch.pfld.pfld import PFLDInference, AuxiliaryNet  
from FacePose_pytorch.compute import find_pose, get_num

from resnet_3d_old.opts import parse_opts
from resnet_3d_old.mean import get_mean, get_std
from resnet_3d_old.model_c import generate_model
from resnet_3d_old.spatial_transforms_winbus import (
    Compose, Normalize, RandomHorizontalFlip, ToTensor, RandomVerticalFlip, 
    ColorAugment)
from resnet_3d_old.temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop

import check_status as cs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取動作種類
def read_classes(file_path):

    fp = open(file_path, "r")
    classes = fp.readline()
    classes = classes. split(",")
    fp.close()

    return classes

#臉部資訊轉換 : 臉部座標資訊作為輸入，轉換成特徵點偵測所需要的輸入格式
def crop_range(x1, x2, y1, y2, w, h):

    size = int(max([w, h]))
    cx = x1 + w/2
    cy = y1 + h/2
    x1 = int(cx - size/2)
    x2 = int(x1 + size)
    y1 = int(cy - size/2)
    y2 = int(y1 + size)

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return x1, x2, y1, y2, dx, dy, edx, edy

# 將影像執行對應的縮放與填補的功能
def letterbox(img, resize_size, mode='square'):

    shape = [img.size[1],img.size[0]]  # current shape [height, width]
    new_shape = resize_size
    if isinstance(new_shape, int):
        ratio = float(new_shape) / max(shape)
    else:
        ratio = max(new_shape) / max(shape)  # ratio  = new / old
    ratiow, ratioh = ratio, ratio
    new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))
    if mode == 'auto':  # minimum rectangle
        dw = np.mod(new_shape - new_unpad[0], 32) / 2  # width padding
        dh = np.mod(new_shape - new_unpad[1], 32) / 2  # height padding
    elif mode == 'square':  # square
        dw = (new_shape - new_unpad[0]) / 2  # width padding
        dh = (new_shape - new_unpad[1]) / 2  # height padding
    elif mode == 'rect':  # square
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding
    elif mode == 'scaleFill':
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape, new_shape)
        ratiow, ratioh = new_shape / shape[1], new_shape / shape[0]

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = img.resize(new_unpad,PIL.Image.ANTIALIAS)
    img = ImageOps.expand(img, border=(left,top,right,bottom), fill=(128))

    return img

# 定義圖片縮放資訊
class letter_img(transforms.Resize):

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    def __call__(self, img):
        return letterbox(img, self.size)
    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

# 將10張影像重組資訊，轉換成對應格式，打包過後的內容會複製堆疊成(1, 0, 2, 3)數量，供模型使用
# spatial_transform : 於主程式當中宣告，用以打包圖像空間資訊
def get_test_data(images, spatial_transform):

    clip = [Image.fromarray(img) for img in images]
    clip = [img.convert('L') for img in clip]
    spatial_transform.randomize_parameters()
    clip = [spatial_transform(img) for img in clip]
    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
    clip = torch.stack((clip,), 0)

    return clip

# 使用此模型預測動作種類，test_data為經過get_test_data()轉換的資料
def predict(model, test_data):

    inputs = Variable(test_data).cuda()
    outputs = model(inputs) 
    outputs = F.softmax(outputs,dim=1)

    return classes[outputs.argmax()]

# GUI選檔介面，選取測試影片，回傳影片路徑
class Qt(QWidget):

    def mv_Chooser(self):

        opt = QFileDialog.Options()
        opt |= QFileDialog.DontUseNativeDialog
        fileUrl = QFileDialog.getOpenFileName(self,"Input Video", "./test_video","Mp4 (*.mp4)", options=opt)
    
        return fileUrl[0]

if __name__ == '__main__':

    # 讀取動作種類

    classes = read_classes('classes.txt')
    print(classes)

    qt_env = QApplication(sys.argv)
    process = Qt()
    fileUrl = process.mv_Chooser()

    if(fileUrl == ""):
        print("Without input file!!")
        sys.exit(0)

    print(fileUrl)

    # 頭部姿態變化狀態
    left_right = ""
    up_down = "" 
    tilt = ""

    # 駕駛行為狀態
    distract_output = ""
    full_clip = []      #暫存3DResnet圖片，駕駛行為動作分類用

    # 綜合危險值
    distract_score = 0

    font = cv2.FONT_HERSHEY_SIMPLEX     # opencv顯示字體


    cg_opt = TestOptions().parse()  # get test options
    cg_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    cg_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    cg_model = create_model(cg_opt)      # create a model given opt.model and other options
    cg_model.setup(cg_opt)               # regular setup: load and print networks; create schedulers
    cg_transform = get_transform(cg_opt)
    if cg_opt.eval:
        cg_model.eval()

    # ----------------------------------------------------

# 駕駛行為分析(動作預測) (使用distract/resnet_3d_old/ 內的程式)

    # 載入參數
    opt = parse_opts()
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    #opt.resume_path = './3dresnet_gray_model.pth'
    opt.resume_path = './best.pth'
    opt.model_depth = 50
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        print('mean:', opt.mean)
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    # 載入模型
    model, parameters = generate_model(opt)
    checkpoint = torch.load(opt.resume_path)
    opt.arch == checkpoint['arch']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    spatial_transform = Compose([
        letter_img(opt.sample_size),
        ToTensor(opt.norm_value), 
        norm_method
    ])

# -------------------------------------------------------------------------------
#頭部姿態分析

    # 載入臉部偵測模型 
    face_model = AntiSpoofPredict(0) # (使用distract/FacePose_pytorch/ 內的程式)

    # 載入特徵點偵測模型 # (使用distract/FacePose_pytorch/ 內的程式)
    headpose_model = './FacePose_pytorch/checkpoint/snapshot/checkpoint.pth.tar'
    checkpoint_h = torch.load(headpose_model, map_location=device)
    plfd_backbone = PFLDInference().to(device)
    plfd_backbone.load_state_dict(checkpoint_h['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.to(device)
    headpose_transformer = transforms.Compose([transforms.ToTensor()])

# -------------------------------------------------------------------------------

    cap = cv2.VideoCapture(fileUrl)
    ret, frame = cap.read()

    height, width = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 儲存結果影片
    videoWriter = cv2.VideoWriter("./result.avi",cv2.VideoWriter_fourcc('X','V','I','D'),fps,(width*2,height))

    while(ret):

        start = time.time()

        cg_start = time.time()
        ret, frame = cap.read()
        if(not ret):
            break
        draw_mat = frame.copy()

        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        data = {'A': cg_transform(Image.fromarray(gray_frame)).unsqueeze(0), 'A_paths': ['doesnt_really_matter']} 

        cg_model.set_input(data)  # unpack data from data loader
        cg_model.test()           # run inference
        visuals = cg_model.get_current_visuals()  # get image results

        im_data = list(visuals.items())[1][1] # grabbing the important part of the result
        cg_im = tensor2im(im_data)  # convert tensor to image
        cg_end = time.time()

        #frame = cv2.resize(im,(512,512))
# -------------------------------------------------------------------------------
# 尋找臉部範圍資訊

        face_start = time.time()
        image_bbox = face_model.get_bbox(frame)
        face_x1 = image_bbox[0]
        face_y1 = image_bbox[1]
        face_x2 = image_bbox[0] + image_bbox[2]
        face_y2 = image_bbox[1] + image_bbox[3]
        face_w = face_x2 - face_x1
        face_h = face_y2 - face_y1
        face_end = time.time()

# -------------------------------------------------------------------------------
# 尋找臉部特徵點

        landmarks_strat = time.time()

        # 計算相關臉部參數
        crop_x1, crop_x2, crop_y1, crop_y2, dx, dy, edx, edy = crop_range(face_x1, face_x2, face_y1, face_y2, face_w, face_h)
        
        # 從原始影像裁切臉部區域   
        cropped = frame[int(crop_y1):int(crop_y2), int(crop_x1):int(crop_x2)]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)
        ratio_w = face_w / 112
        ratio_h = face_h / 112

        # 縮放影像112x112
        cropped = cv2.resize(cropped, (112, 112))
        face_input = cropped.copy()
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = headpose_transformer(face_input).unsqueeze(0).to(device)

        # 預測特徵點位置
        _, landmarks = plfd_backbone(face_input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [112, 112]

        point_dict = {}
        i = 0
        # 繪製特徵點
        for (x,y) in pre_landmark.astype(np.float32):
            point_dict[f'{i}'] = [x,y]
            cv2.circle(draw_mat,(int(face_x1 + x * ratio_w),int(face_y1 + y * ratio_h)), 2, (255, 0, 0), -1)
            i += 1

        landmarks_end = time.time()
        cv2.circle(draw_mat,(int(face_x1 + get_num(point_dict, 1, 0) * ratio_w),int(face_y1 + get_num(point_dict, 1, 1) * ratio_h)), 2, (255, 0, 0), -1)
        cv2.circle(draw_mat,(int(face_x1 + get_num(point_dict, 31, 0) * ratio_w),int(face_y1 + get_num(point_dict, 31, 1) * ratio_h)), 2, (255, 0, 0), -1)
        cv2.circle(draw_mat,(int(face_x1 + get_num(point_dict, 51, 0) * ratio_w),int(face_y1 + get_num(point_dict, 51, 1) * ratio_h)), 2, (255, 0, 0), -1)
# -------------------------------------------------------------------------------
# 頭部姿態
        
        # 計算頭部三軸角度
        yaw, pitch, roll = find_pose(point_dict)

        # 偵測單一圖片頭部姿態
        # left_right, up_down, tilt = cs.headpose.headpose_status(yaw, pitch, roll)

        cs.headpose.headpose_series(yaw, pitch, roll)

# -------------------------------------------------------------------------------
# 分心偵測
        
        # 暫存影像供駕駛行為分析使用(動作分類)
        #gray_frame = cv2.resize(gray_frame, (224,224))
        #full_clip.append(gray_frame)
        cg_im = cv2.resize(cg_im, (224,224))
        full_clip.append(cg_im)

        distract_start = 0
        distract_end = 0
        headmotion_start = 0
        headmotion_end = 0

        # 累計滿10張影像, 累計滿10筆頭部姿態資料
        if len(full_clip) > 9:

            # ------------------------------------------------------------
            # 駕駛行為分析
            distract_start = time.time()
            test_data = get_test_data(full_clip, spatial_transform)         
            distract_output = predict(model, test_data)
            distract_end = time.time()

            # ------------------------------------------------------------
            # 頭部姿態變化分析
            headmotion_start = time.time()

            left_right, up_down, tilt = cs.headpose.headpose_output()
            if(face_w < 20 or face_h < 20):
                left_right, up_down, tilt = "", "", ""
            headmotion_end = time.time()

            # ------------------------------------------------------------
            # 計算綜合危險值
            distract_score = cs.dis_head(distract_output, left_right, up_down, tilt)

            #------------------------------------------------------------
            # 清除暫存資料
            cs.headpose.clear()
            full_clip = []


        end = time.time()

        
        """
        print("#####################################################################")
        print("\n")
        print("--------------處理時間--------------")
        print('Cycle GAN    : ', round(cg_end-cg_start,3), '(s)')
        print('臉部偵測     : ', round(face_end-face_start,3), '(s)')
        print('特徵點偵測   : ', round(landmarks_end-landmarks_strat,3), '(s)')
        print('駕駛行為分析 : ', round(distract_end-distract_start,3), '(s)')
        print('頭部姿態分析 : ', round(headmotion_end-headmotion_start,3), '(s)')
        print("\n")
        print("--------------預測資訊--------------")
        print('駕駛行為分析 : ', distract_output)
        print('Yaw角度      : ', yaw)
        print('Pitch角度    : ', pitch)
        print('Roll角度     : ', roll)
        print('左右轉動     : ', left_right)
        print('上下仰俯     : ', up_down)
        print('左右歪斜     : ', tilt)
        print('危險值       : ', distract_score)
        print("\n")
        print("----------------警示----------------")
        if(distract_score >= 35):
            print('Yes')
        else:
            print('No')

        print("\n\n")
        """
        
# -------------------------------------------------------------------------------
# 繪製畫面      
        cv2.rectangle(draw_mat, (0, 0), (230, 230), (255, 255, 255), -1, cv2.LINE_AA)

        cv2.rectangle(draw_mat, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 255), 2, cv2.LINE_AA)

        cv2.putText(draw_mat,"R-L",(10,65), font,0.6,(255,0,0),2) 
        cv2.putText(draw_mat,"U-D",(10,95), font,0.6,(255,0,0),2)
        cv2.putText(draw_mat,"TILT",(10,125), font,0.6,(255,0,0),2)

        cv2.putText(draw_mat, ": "+str(left_right), (75,65), font,0.8,(255,0,0),2)
        cv2.putText(draw_mat, ": "+str(up_down), (75,95), font,0.8,(255,0,0),2)
        cv2.putText(draw_mat, ": "+str(tilt), (75,125), font,0.8,(255,0,0),2)
        
        cv2.putText(draw_mat,"Status",(10,35), font,0.6,(255,0,0),2) 
        cv2.putText(draw_mat,": "+distract_output,(75,35), font, 0.8,(255,0,0),2)
        
        cv2.putText(draw_mat,"FPS : "+str(int(1/(end-start)+0.000001)),(10,155), font, 0.8,(0,0,0),2)
        

        if(distract_score >= 35):
            cv2.rectangle(draw_mat, (10, 170), (220, 225), (120, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(draw_mat,"dangerous!",(20,210), font, 1.1,(120,0,255),2,cv2.LINE_AA)
        
# -------------------------------------------------------------------------------
        cg_im = cv2.resize(cg_im, (width,height))
        cv2.putText(cg_im,"Cycle GAN",(30,30), font, 1,(0,0,255),2)
        image_c = cv2.hconcat([draw_mat, cg_im])
        #cv2.imshow("cyclegan", cg_im)
        #cv2.imshow("frame", draw_mat)
        cv2.imshow("image_c", image_c)
        videoWriter.write(image_c)

        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            videoWriter.release()
            break
        
    videoWriter.release()
    cap.release()
