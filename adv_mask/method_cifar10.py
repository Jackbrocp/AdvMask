import random
import numpy as np
from PIL import Image
from adv_mask import gridmask
import copy
class PointChoose():
    def __init__(self):
        self.point_list =  ''
        self.width = 32
        self.height = 32
        self.rotate = 1
        self.translation = 2
        self.gridmask_num = 0
    def visualize(self,mask,name):
        img = np.ones((mask.shape))*255
        img = img*mask
        img = Image.fromarray(np.uint8(img))
        img.save('./'+name+'.png')
    def createMask(self,mask_num,pointlist):
        self.point_list = pointlist
        random.shuffle(self.point_list)
        if len(self.point_list)==0:
            self.gridmask_num += 1
            grid = gridmask.Grid(d1=24,d2=33,rotate=1,ratio=0.4,mode=1,prob=1.)
            mask = grid()
            return mask
        ########################################################
        length_scope=[5,20]
        length = random.randint(length_scope[0],length_scope[1]+1)
        gap=1
        direction=[-1,1]
        self.translation = length_scope[1]//2+gap-1
        vertical = np.random.randint(self.translation)*direction[np.random.randint(2)]
        horizontal = np.random.randint(self.translation)*direction[np.random.randint(2)]
        mask_scope=[0.03,0.5]
        conflict_ratio = 0.10
        mask = np.ones((self.width+length_scope[1]//2*2+2*gap,self.height+length_scope[1]//2*2+2*gap),np.float32)
        index = random.sample(range(0,len(self.point_list)),len(self.point_list))
        i=0
        point_num=0  
        mask_ratio=0
        while i<len(index) and mask_ratio<mask_scope[0]:
            x,y = self.point_list[i]
            x+=length_scope[1]//2+gap
            y+=length_scope[1]//2+gap
            x0 = x -(length-1)//2
            y0 = y-(length-1)//2
            area = mask[x0-gap:x0+length+gap,y0-gap:y0+length+gap]
            if (area==0).sum()>int(length**2*conflict_ratio):
                i+=1
                continue
            mask[x0:x+(length-1)//2,y0:y+(length-1)//2]*=0
            mask_ratio = (mask[length_scope[1]//2+gap:length_scope[1]//2+gap+self.width,length_scope[1]//2+gap:length_scope[1]//2+gap+self.height]==0).sum()/(self.width*self.height)
            if mask_ratio>=mask_scope[1]:
                break
            i+=1
            point_num+=1
        
        r = np.random.randint(self.rotate)
        background = Image.fromarray(np.uint8(mask))
        background = background.rotate(r)
        background = np.array(background)
        x0 = length_scope[1]//2+gap+vertical
        y0 = length_scope[1]//2+gap+horizontal
        mask = background[x0 :x0+self.width ,y0 :y0+self.height ]
        if (mask==0).sum()<100:
            self.gridmask_num += 1
            grid = gridmask.Grid(d1=24,d2=33,rotate=1,ratio=0.4,mode=1,prob=1.)
            mask = grid()
            return mask 
        mask_num.append( (mask==0).sum())
        return mask 


 